"""MediaPipe Face Landmarker wrapper.

MediaPipe moved from the `solutions` API to the `tasks` API during 0.10.x, and
which one a given wheel exposes varies by version and platform. Both are
supported here and the backend is chosen at runtime, so the pipeline does not
break when the environment is rebuilt with a different version.

The tasks API needs a model file. It is fetched once to a local cache; pass
--model to use a copy you already have, which is what you want on a machine
with no outbound access.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_DIR = Path(
    os.environ.get("DROWSY_MODEL_DIR", Path.home() / ".cache" / "drowsy_pipeline")
)


def ensure_model(path: Path | None = None) -> Path:
    """Return a local face_landmarker.task, downloading it once if needed."""
    target = Path(path) if path else DEFAULT_MODEL_DIR / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[landmarks] downloading face landmarker model to {target}")
    # Into a private temp file, then renamed: shards start together, and one seeing
    # another's half-written model would take it as complete because it is non-empty.
    partial = target.with_name(f"{target.name}.{os.getpid()}.part")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(target)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"could not download the landmarker model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and pass it with --model <path>."
        ) from exc
    return target


class FaceLandmarker:
    """Detect 478 face landmarks per frame, in pixel coordinates.

    Usage:
        with FaceLandmarker() as fl:
            pts = fl(bgr_frame)      # (478, 3) float array, or None

    The returned array is in pixels for x and y so the geometry module can work
    in a single coordinate system; z keeps MediaPipe's relative depth units.
    """

    def __init__(self, model_path: Path | None = None, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._backend = None
        self._kind = None
        self._init_backend(model_path)

    def _init_backend(self, model_path: Path | None) -> None:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._kind = "solutions"
            self._backend = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        resolved = ensure_model(model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(resolved)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._kind = "tasks"
        self._backend = vision.FaceLandmarker.create_from_options(opts)

    @property
    def backend(self) -> str:
        return self._kind or "uninitialised"

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> np.ndarray | None:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._kind == "solutions":
            res = self._backend.process(rgb)
            faces = getattr(res, "multi_face_landmarks", None)
            if not faces:
                return None
            lms = faces[0].landmark
            return np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=float)

        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._backend.detect_for_video(image, timestamp_ms)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        return np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=float)

    def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
        self._backend = None

    def __enter__(self) -> "FaceLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
