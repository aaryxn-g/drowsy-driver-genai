"""Blink / PERCLOS tests on synthetic openness series with known ground truth."""
from __future__ import annotations

import numpy as np
import pytest

from src import blinks, config

FPS = 30.0


def series_with_dips(n_frames: int, dips: list[tuple[int, int]],
                     floor: float = 0.05, ceiling: float = 1.0) -> np.ndarray:
    """Open series of `n_frames` with closures at (start_frame, n_frames).

    Square dips, so the intended duration equals the measured duration exactly.
    A real eyelid ramps over a frame or two at each end; the asymmetric ramp
    that matters for drowsiness is exercised separately in the reopening test.
    """
    out = np.full(n_frames, ceiling, dtype=float)
    for start, length in dips:
        out[start:start + length] = floor
    return out


class TestBaseline:
    def test_percentiles(self):
        ear = np.concatenate([np.full(900, 0.30), np.full(100, 0.05)])
        b = blinks.estimate_baseline(ear, source="alert")
        assert b.ear_open == pytest.approx(0.30, abs=1e-6)
        assert b.ear_closed < 0.10
        assert b.span > 0.15

    def test_rejects_too_few_frames(self):
        with pytest.raises(ValueError):
            blinks.estimate_baseline(np.full(5, 0.3))

    def test_nans_ignored(self):
        ear = np.concatenate([np.full(500, 0.3), np.full(500, np.nan)])
        assert blinks.estimate_baseline(ear).n_frames == 500


class TestOpenness:
    def test_maps_baseline_to_unit_interval(self):
        b = blinks.EyeBaseline(ear_open=0.30, ear_closed=0.05, source="t", n_frames=100)
        got = blinks.openness(np.array([0.05, 0.175, 0.30]), b)
        assert got == pytest.approx([0.0, 0.5, 1.0])

    def test_preserves_nan(self):
        b = blinks.EyeBaseline(ear_open=0.30, ear_closed=0.05, source="t", n_frames=100)
        assert np.isnan(blinks.openness(np.array([np.nan]), b))[0]

    def test_different_eye_shapes_normalise_alike(self):
        """A narrow-eyed and a wide-eyed subject, both half shut, agree."""
        narrow = blinks.EyeBaseline(0.18, 0.02, "t", 100)
        wide = blinks.EyeBaseline(0.34, 0.04, "t", 100)
        a = blinks.openness(np.array([0.02 + 0.5 * 0.16]), narrow)[0]
        b = blinks.openness(np.array([0.04 + 0.5 * 0.30]), wide)[0]
        assert a == pytest.approx(b)


class TestClosureDetection:
    def test_counts_and_classifies_by_duration(self):
        # 0.2 s blink, 0.7 s long closure, 1.5 s microsleep
        s = series_with_dips(600, [(50, 6), (200, 21), (400, 45)])
        ev = blinks.detect_closures(s, FPS)
        assert len(ev) == 3
        assert [e.kind for e in ev] == ["blink", "long_closure", "microsleep"]

    def test_durations_are_accurate(self):
        s = series_with_dips(400, [(100, 9)])
        ev = blinks.detect_closures(s, FPS)
        assert len(ev) == 1
        assert ev[0].duration_s == pytest.approx(9 / FPS, abs=1.0 / FPS)

    def test_open_eye_yields_no_events(self):
        assert blinks.detect_closures(np.full(600, 1.0), FPS) == []

    def test_hysteresis_rejects_threshold_noise(self):
        """A signal sagging near `enter` but never closing is not a closure."""
        rng = np.random.default_rng(0)
        noisy = 0.46 + rng.normal(0, 0.01, 600)
        assert len(blinks.detect_closures(noisy, FPS)) == 0

    def test_short_flicker_below_min_frames_ignored(self):
        s = np.full(300, 1.0)
        s[100] = 0.1  # single frame
        assert blinks.detect_closures(s, FPS) == []

    def test_reopening_slower_than_closing_is_measurable(self):
        """Drowsy eyes reopen slowly. The asymmetry must show in the speeds."""
        s = np.full(300, 1.0)
        s[100:106] = np.linspace(1.0, 0.05, 6)    # fast close, 6 frames
        s[106:136] = np.linspace(0.05, 1.0, 30)   # slow reopen, 30 frames
        ev = blinks.detect_closures(s, FPS)
        assert len(ev) == 1
        assert ev[0].closing_speed > ev[0].reopening_speed

    def test_lost_tracking_truncates_rather_than_inventing(self):
        s = np.full(300, 1.0)
        s[100:110] = 0.1
        s[105:110] = np.nan
        ev = blinks.detect_closures(s, FPS)
        assert len(ev) == 1
        assert ev[0].truncated


class TestPerclos:
    def test_fraction_below_threshold(self):
        """Half the window closed gives PERCLOS 0.5."""
        s = np.concatenate([np.full(900, 0.0), np.full(900, 1.0)])
        out = blinks.perclos_series(s, FPS, window_s=60.0, step_s=60.0)
        assert out["perclos"][0] == pytest.approx(0.5)

    def test_fully_open_is_zero(self):
        out = blinks.perclos_series(np.full(1800, 1.0), FPS, window_s=60, step_s=60)
        assert out["perclos"][0] == pytest.approx(0.0)

    def test_normal_blinking_stays_far_below_drowsy(self):
        """20 blinks/min must not look like a driver nodding off.

        Square dips hold the eye fully shut for the whole blink, so 0.10 here
        is the worst case; a real 0.3 s blink is >80% closed for only part of
        its span, which is why alert PERCLOS sits nearer 0.03 in practice.
        What matters is the separation from sustained closure.
        """
        blinking = series_with_dips(
            int(60 * FPS), [(int(i * 1.5 * FPS), 9) for i in range(20)]
        )
        drowsy = np.full(int(60 * FPS), 1.0)
        drowsy[: int(20 * FPS)] = 0.05

        kw = dict(fps=FPS, window_s=60.0, step_s=60.0)
        p_blink = blinks.perclos_series(blinking, **kw)["perclos"][0]
        p_drowsy = blinks.perclos_series(drowsy, **kw)["perclos"][0]

        assert p_blink <= 0.10
        assert p_drowsy > 3 * p_blink

    def test_sustained_closure_scores_high(self):
        s = np.full(int(60 * FPS), 1.0)
        s[: int(20 * FPS)] = 0.05  # 20 s shut out of 60
        out = blinks.perclos_series(s, FPS, window_s=60.0, step_s=60.0)
        assert out["perclos"][0] == pytest.approx(1 / 3, abs=0.02)

    def test_missing_frames_do_not_count_as_closed(self):
        """A tracking failure must not be reported as a drowsy driver."""
        s = np.full(1800, 1.0)
        s[:900] = np.nan
        out = blinks.perclos_series(s, FPS, window_s=60.0, step_s=60.0)
        assert out["perclos"][0] == pytest.approx(0.0)
        assert out["coverage"][0] == pytest.approx(0.5)

    def test_all_missing_window_is_nan_not_zero(self):
        out = blinks.perclos_series(np.full(1800, np.nan), FPS, window_s=60, step_s=60)
        assert np.isnan(out["perclos"][0])
        assert out["coverage"][0] == pytest.approx(0.0)

    def test_window_slides(self):
        out = blinks.perclos_series(np.full(int(120 * FPS), 1.0), FPS,
                                    window_s=60.0, step_s=10.0)
        assert len(out["t_s"]) > 5
        assert np.all(np.diff(out["t_s"]) > 0)


class TestSummary:
    def test_alert_and_drowsy_separate(self):
        """End-to-end: the summary must rank a drowsy clip above an alert one."""
        base = blinks.EyeBaseline(0.30, 0.05, "alert", 1000)
        n = int(120 * FPS)

        alert = series_with_dips(n, [(int(i * 3 * FPS), 9) for i in range(40)])
        drowsy = series_with_dips(n, [(int(i * 3 * FPS), 45) for i in range(40)])

        zeros = np.zeros(n)
        a = blinks.summarise(alert, zeros, zeros, FPS, base)
        d = blinks.summarise(drowsy, zeros, zeros, FPS, base)

        assert d["perclos80_mean"] > a["perclos80_mean"]
        assert d["n_microsleeps"] > a["n_microsleeps"]
        assert a["n_microsleeps"] == 0
        assert a["blink_rate_per_min"] > 0

    def test_flags_low_detection_rate(self):
        base = blinks.EyeBaseline(0.30, 0.05, "alert", 1000)
        s = np.full(1800, 1.0)
        s[:1200] = np.nan
        out = blinks.summarise(s, np.zeros(1800), np.zeros(1800), FPS, base)
        assert out["low_quality"]
        assert out["detection_rate"] == pytest.approx(1 / 3, abs=0.01)
