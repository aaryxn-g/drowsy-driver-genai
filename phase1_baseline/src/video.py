"""Open a video the right way up, and record what it took to get there.

UTA-RLDD was filmed on phones, so many files store sideways pixels plus a rotation
flag in the container. A sideways face is not a loud failure. EAR is vertical over
horizontal, so a 90 degree turn swaps its axes and a blink pushes EAR up instead of
down. The subject's baseline comes from their alert video and is reused across all
three, so one such file corrupts a whole subject without raising anything.

Whether cv2 acts on the flag depends on the build, so nothing here assumes it does.
AUTO is requested, the first decoded frame is checked against a prediction, and
frames are rotated by hand if AUTO was ignored. Three traps shape this module:

* CAP_PROP_FRAME_WIDTH/HEIGHT report the ROTATED size while AUTO is on. Predicting
  from them with AUTO on swaps twice, decides AUTO failed, and rotates a frame that
  is already upright. The stored size is therefore read with AUTO off.
* A 180 degree turn, or a square frame, has the same shape either way, so shape
  cannot show whether AUTO was applied. Those cases compare pixels instead.
* Only the FFmpeg backend reports the flag. Another backend answers 0, which looks
  exactly like an unrotated file, so FFmpeg is requested explicitly.

SIGN CONVENTION: rotation_deg is OpenCV's ORIENTATION_META, the clockwise turn needed
to display the frame. ffprobe prints the opposite sign: its "rotation -90" is
rotation_deg 90 here, and its "+90" is rotation_deg 270.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

# How far to look for a frame that differs from its own rotation, when only pixels
# can settle whether AUTO was applied. A flat frame, like a fade from black, cannot.
PIXEL_CHECK_MAX_FRAMES = 30

_CV2_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class RotationCheckError(RuntimeError):
    """The decoded frames do not match any orientation we can account for."""


def normalise_rotation(degrees: float) -> int:
    """Map any right angle onto 0, 90, 180 or 270. Anything else is refused."""
    rounded = int(round(float(degrees)))
    if abs(rounded - float(degrees)) > 1e-6 or rounded % 90 != 0:
        raise ValueError(f"rotation is not a right angle: {degrees}")
    return rounded % 360


def apply_rotation(frame: np.ndarray, degrees: float) -> np.ndarray:
    """Turn a frame clockwise by `degrees` (OpenCV's sign, see the module docstring)."""
    angle = normalise_rotation(degrees)
    return frame if angle == 0 else cv2.rotate(frame, _CV2_ROTATE[angle])


def predicted_size(stored_wh: tuple[int, int], degrees: float) -> tuple[int, int]:
    """(width, height) once a frame of stored size `stored_wh` has been rotated."""
    width, height = stored_wh
    return (height, width) if normalise_rotation(degrees) % 180 else (width, height)


def classify_rotation(
    actual_wh: tuple[int, int], stored_wh: tuple[int, int], degrees: float
) -> str | None:
    """Decide from the first frame's size who has to rotate the rest.

    Returns "none" for an unrotated video, "auto" if the decoder rotated the frame,
    "manual" if it did not, or None when shape cannot tell (180 degrees or a square
    frame). Raises RotationCheckError when the size matches neither expectation.
    """
    angle = normalise_rotation(degrees)
    predicted = predicted_size(stored_wh, angle)
    if actual_wh == predicted:
        if angle == 0:
            return "none"
        return None if predicted == stored_wh else "auto"
    if actual_wh == stored_wh:
        return "manual"
    raise RotationCheckError(
        f"first frame is {actual_wh}, but stored size {stored_wh} with rotation "
        f"{angle} predicts {predicted}"
    )


def downscale(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, int(round(h * max_width / w))),
                      interpolation=cv2.INTER_AREA)


def is_low_resolution(width: float, height: float) -> bool:
    """Flag on the short side, which is the same whichever way up the video is."""
    return bool(min(width, height) < config.MIN_RESOLUTION_SHORT_SIDE)


@dataclass(frozen=True)
class VideoInfo:
    """Per-video quality covariates, in the columns the manifest and log record."""
    rotation_deg: int
    rotation_method: str      # none, auto or manual
    stored_width: int
    stored_height: int
    effective_width: int      # after rotation, before any max_width downscale
    effective_height: int
    fps: float
    duration_s: float
    bitrate_kbps: float

    def as_row(self) -> dict:
        return asdict(self)


INFO_COLUMNS = list(VideoInfo.__dataclass_fields__)


def _settle_by_pixels(path: Path, angle: int) -> str:
    """Compare AUTO on against AUTO off when shape cannot separate them."""
    raw = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    auto = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        raw.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        auto.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        for _ in range(PIXEL_CHECK_MAX_FRAMES):
            ok_raw, frame_raw = raw.read()
            ok_auto, frame_auto = auto.read()
            if not (ok_raw and ok_auto):
                break
            turned = apply_rotation(frame_raw, angle)
            if np.array_equal(turned, frame_raw):
                continue  # this frame looks the same either way, try the next
            if np.array_equal(frame_auto, turned):
                return "auto"
            if np.array_equal(frame_auto, frame_raw):
                return "manual"
            raise RotationCheckError(
                f"{path}: frame with AUTO on is neither the raw frame nor the raw "
                f"frame rotated by {angle}")
    finally:
        raw.release()
        auto.release()
    raise RotationCheckError(
        f"{path}: rotation {angle} and no frame in the first "
        f"{PIXEL_CHECK_MAX_FRAMES} shows whether it was applied")


class OrientedVideo:
    """Frames the right way up, with the evidence for it in `.info`.

    with OrientedVideo(path) as video:
        while (frame := video.read()) is not None: ...

    Raises instead of guessing: a video that fails here is reported as failed, which
    is recoverable, where a sideways one would quietly poison a subject's baseline.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        try:
            self._first = self._inspect()
        except Exception:
            self.release()
            raise

    def _inspect(self) -> np.ndarray:
        cap = self._cap
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) == 1:
            raise RotationCheckError(
                f"{self.path}: cannot turn AUTO off, so the stored size is unknowable")
        stored = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        angle = normalise_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)

        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError(f"no decodable frames: {self.path}")
        h, w = first.shape[:2]
        method = classify_rotation((w, h), stored, angle)
        if method is None:
            method = _settle_by_pixels(self.path, angle)

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 240:
            fps = 30.0  # some phone-recorded files report nonsense
        n_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n_declared / fps if n_declared > 0 else float("nan")
        bitrate = cap.get(cv2.CAP_PROP_BITRATE) or 0.0
        if bitrate <= 0 and duration > 0:
            bitrate = self.path.stat().st_size * 8 / duration / 1000.0

        effective = predicted_size(stored, angle)
        self.n_frames_declared = n_declared
        self.info = VideoInfo(
            rotation_deg=angle,
            rotation_method=method,
            stored_width=stored[0],
            stored_height=stored[1],
            effective_width=effective[0],
            effective_height=effective[1],
            fps=fps,
            duration_s=duration,
            bitrate_kbps=bitrate if bitrate > 0 else float("nan"),
        )
        self._to_apply = angle if method == "manual" else 0
        return first

    def read(self) -> np.ndarray | None:
        """The next upright frame at full resolution, or None at the end."""
        if self._first is not None:
            frame, self._first = self._first, None
        else:
            ok, frame = self._cap.read()
            if not ok:
                return None
        frame = apply_rotation(frame, self._to_apply)
        h, w = frame.shape[:2]
        expected = (self.info.effective_width, self.info.effective_height)
        if (w, h) != expected:
            raise RotationCheckError(
                f"{self.path}: frame is {(w, h)} after rotation, expected {expected}")
        return frame

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "OrientedVideo":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def probe(path: Path) -> VideoInfo:
    """Read a video's orientation and quality covariates without processing it."""
    with OrientedVideo(path) as video:
        return video.info
