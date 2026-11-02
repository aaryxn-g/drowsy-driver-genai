"""Eye-closure dynamics: openness normalisation, blink events, PERCLOS.

This is where a blink gets separated from drowsy closure. A single frame cannot
tell them apart: both show a shut eye. The difference is entirely temporal.

  normal blink      ~100-400 ms, sharp V, fast reopening
  prolonged closure ~0.4-1.0 s,  slower reopening
  microsleep        >1 s

PERCLOS sidesteps event classification altogether. It measures the fraction of
time the eye is at least 80% closed over a rolling window, so ordinary blinking
contributes a small, roughly constant background while sustained closure
dominates. Both views are computed here because they fail in different ways.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config


# --------------------------------------------------------------- normalisation

@dataclass(frozen=True)
class EyeBaseline:
    """A subject's fully-open and fully-closed EAR, in EAR units."""
    ear_open: float
    ear_closed: float
    source: str          # which video the baseline was estimated from
    n_frames: int

    @property
    def span(self) -> float:
        return self.ear_open - self.ear_closed


def estimate_baseline(ear: np.ndarray, source: str = "") -> EyeBaseline:
    """Estimate per-subject open/closed EAR from a series of EAR values.

    Use the subject's ALERT recording for this. A drowsy recording may never
    contain a fully open eye, so a percentile taken within it would rescale the
    drowsiness away and make every subject look equally alert. That failure is
    silent, which is what makes it dangerous.
    """
    valid = ear[np.isfinite(ear)]
    if valid.size < 30:
        raise ValueError(f"not enough valid EAR frames to set a baseline: {valid.size}")
    return EyeBaseline(
        ear_open=float(np.percentile(valid, config.EAR_OPEN_PERCENTILE)),
        ear_closed=float(np.percentile(valid, config.EAR_CLOSED_PERCENTILE)),
        source=source,
        n_frames=int(valid.size),
    )


def openness(ear: np.ndarray, baseline: EyeBaseline) -> np.ndarray:
    """Rescale EAR to [0, 1] against a subject baseline.

    0 = as closed as this subject's eye gets, 1 = fully open. Values are clipped
    to [0, 1.5]: a little headroom above 1 keeps genuine wide-eyed frames from
    being flattened, while NaN is preserved so missing detections stay missing.
    """
    if baseline.span < 1e-6:
        raise ValueError("degenerate eye baseline: open and closed EAR coincide")
    out = (ear - baseline.ear_closed) / baseline.span
    return np.clip(out, 0.0, 1.5)


# ------------------------------------------------------------- closure events

@dataclass
class ClosureEvent:
    start_frame: int
    end_frame: int
    start_s: float
    duration_s: float
    min_openness: float
    closing_speed: float      # openness units per second
    reopening_speed: float    # openness units per second
    kind: str                 # blink | long_closure | microsleep
    truncated: bool           # detection was lost mid-event


def _classify(duration_s: float) -> str:
    if duration_s <= config.BLINK_MAX_S:
        return "blink"
    if duration_s <= config.LONG_CLOSURE_MAX_S:
        return "long_closure"
    return "microsleep"


def detect_closures(
    open_series: np.ndarray,
    fps: float,
    enter: float = config.EVENT_ENTER,
    exit_: float = config.EVENT_EXIT,
    min_frames: int = config.MIN_EVENT_FRAMES,
    min_depth: float = config.EVENT_MIN_DEPTH,
) -> list[ClosureEvent]:
    """Find eye-closure events with hysteresis.

    An event starts when openness drops below `enter` and ends when it climbs
    back above `exit_`. The gap between the two thresholds prevents a signal
    hovering near the boundary from registering a burst of phantom blinks, and
    `min_depth` discards events where the eye never actually got close to shut.

    A run of missing detections inside an event ends it and marks it truncated,
    because we cannot know what the eye did while the face was not tracked.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    events: list[ClosureEvent] = []
    in_event = False
    start = 0
    truncated = False

    def close_event(end: int) -> None:
        nonlocal in_event, truncated
        in_event = False
        seg = open_series[start:end + 1]
        finite = seg[np.isfinite(seg)]
        if end - start + 1 < min_frames or finite.size == 0:
            truncated = False
            return
        rel_min = int(np.nanargmin(seg))
        min_val = float(seg[rel_min])
        if min_val >= min_depth:
            truncated = False
            return
        duration_s = (end - start + 1) / fps
        closing = (
            (float(seg[0]) - min_val) / (rel_min / fps) if rel_min > 0 else float("nan")
        )
        tail = len(seg) - 1 - rel_min
        reopening = (
            (float(seg[-1]) - min_val) / (tail / fps) if tail > 0 else float("nan")
        )
        events.append(
            ClosureEvent(
                start_frame=start,
                end_frame=end,
                start_s=start / fps,
                duration_s=duration_s,
                min_openness=min_val,
                closing_speed=closing,
                reopening_speed=reopening,
                kind=_classify(duration_s),
                truncated=truncated,
            )
        )
        truncated = False

    for i, v in enumerate(open_series):
        if not np.isfinite(v):
            if in_event:
                truncated = True
                close_event(i - 1)
            continue
        if not in_event and v < enter:
            in_event = True
            start = i
            truncated = False
        elif in_event and v > exit_:
            close_event(i - 1)

    if in_event:
        truncated = True
        close_event(len(open_series) - 1)
    return events


# -------------------------------------------------------------------- PERCLOS

def perclos_series(
    open_series: np.ndarray,
    fps: float,
    window_s: float = config.PERCLOS_WINDOW_S,
    step_s: float = config.PERCLOS_STEP_S,
    threshold: float = config.PERCLOS_P80_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Rolling PERCLOS.

    Returned per window:
      t_s      window centre, seconds
      perclos  fraction of VALID frames with openness below threshold
      coverage fraction of frames in the window that were valid at all

    Missing detections are excluded from the numerator and the denominator, not
    counted as closed. Counting them as closed would make a tracking failure
    look like a drowsy driver. `coverage` is returned so a window computed from
    very little data can be discarded rather than trusted.
    """
    win = max(1, int(round(window_s * fps)))
    step = max(1, int(round(step_s * fps)))
    n = len(open_series)
    if n == 0:
        empty = np.array([], dtype=float)
        return {"t_s": empty, "perclos": empty.copy(), "coverage": empty.copy()}

    centres, values, coverages = [], [], []
    for start in range(0, max(1, n - win + 1), step):
        seg = open_series[start:start + win]
        finite = np.isfinite(seg)
        n_valid = int(finite.sum())
        coverages.append(n_valid / len(seg))
        if n_valid == 0:
            values.append(float("nan"))
        else:
            values.append(float((seg[finite] < threshold).sum() / n_valid))
        centres.append((start + len(seg) / 2.0) / fps)

    return {
        "t_s": np.asarray(centres, dtype=float),
        "perclos": np.asarray(values, dtype=float),
        "coverage": np.asarray(coverages, dtype=float),
    }


# -------------------------------------------------------------- video summary

def summarise(
    open_series: np.ndarray,
    mar: np.ndarray,
    pitch: np.ndarray,
    fps: float,
    baseline: EyeBaseline,
) -> dict[str, float]:
    """Collapse one video into the features a sequence model consumes.

    Deliberately small and interpretable. These are the quantities the
    drowsiness literature actually validates, not an arbitrary feature dump.
    """
    finite = np.isfinite(open_series)
    n = len(open_series)
    detection_rate = float(finite.sum() / n) if n else 0.0
    duration_s = n / fps if fps > 0 else 0.0

    events = detect_closures(open_series, fps)
    blinks = [e for e in events if e.kind == "blink"]
    longs = [e for e in events if e.kind == "long_closure"]
    micros = [e for e in events if e.kind == "microsleep"]

    p80 = perclos_series(open_series, fps, threshold=config.PERCLOS_P80_THRESHOLD)
    p70 = perclos_series(open_series, fps, threshold=config.PERCLOS_P70_THRESHOLD)

    def _mean(vals: list[float]) -> float:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        return float(arr.mean()) if arr.size else float("nan")

    minutes = duration_s / 60.0 if duration_s > 0 else float("nan")
    valid_mar = mar[np.isfinite(mar)]
    yawn_rate = (
        float((valid_mar > 0.6).sum() / valid_mar.size) if valid_mar.size else float("nan")
    )

    return {
        "duration_s": duration_s,
        "fps": fps,
        "n_frames": n,
        "detection_rate": detection_rate,
        "ear_open_baseline": baseline.ear_open,
        "ear_closed_baseline": baseline.ear_closed,
        "baseline_source": baseline.source,
        "mean_openness": float(np.nanmean(open_series)) if finite.any() else float("nan"),
        "perclos80_mean": float(np.nanmean(p80["perclos"])) if p80["perclos"].size else float("nan"),
        "perclos80_max": float(np.nanmax(p80["perclos"])) if p80["perclos"].size else float("nan"),
        "perclos70_mean": float(np.nanmean(p70["perclos"])) if p70["perclos"].size else float("nan"),
        "n_blinks": len(blinks),
        "blink_rate_per_min": len(blinks) / minutes if minutes and minutes > 0 else float("nan"),
        "mean_blink_duration_s": _mean([e.duration_s for e in blinks]),
        "mean_reopening_speed": _mean([e.reopening_speed for e in blinks]),
        "mean_closing_speed": _mean([e.closing_speed for e in blinks]),
        "n_long_closures": len(longs),
        "n_microsleeps": len(micros),
        "total_closure_s": float(sum(e.duration_s for e in events)),
        "mar_mean": float(np.nanmean(mar)) if np.isfinite(mar).any() else float("nan"),
        "mar_p95": float(np.nanpercentile(mar[np.isfinite(mar)], 95)) if valid_mar.size else float("nan"),
        "yawn_frame_fraction": yawn_rate,
        "pitch_std": float(np.nanstd(pitch)) if np.isfinite(pitch).any() else float("nan"),
        "pitch_range": (
            float(np.nanmax(pitch) - np.nanmin(pitch)) if np.isfinite(pitch).any() else float("nan")
        ),
        "low_quality": detection_rate < config.MIN_DETECTION_RATE,
    }


def events_to_records(events: list[ClosureEvent]) -> list[dict]:
    return [asdict(e) for e in events]
