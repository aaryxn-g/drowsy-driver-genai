"""Stage 5 of the Phase 2 pilot, Gate 4: does reenactment preserve eye closure?

Runs the UNCHANGED phase1_baseline extractor and aggregator (as subprocesses, from that
directory) over the 32 synthetic clips and the 8 driving crops, then compares.

Baseline policy. Phase 1 takes each subject's eye baseline from their alert video. A synthetic
clip has no alert video. For the pilot, each clip's openness is computed against ITS OWN clip
percentiles. This is done without touching Phase 1 code: the aggregator is fed a derived
manifest in which subject_id := video_id, so `build_baselines` pools exactly one clip per
"subject". The pilot measures correlation, not absolute PERCLOS, and a within-clip percentile
compresses drowsy clips, so this policy is conservative for the correlation. Choosing the real
baseline policy for synthetic identities is an OPEN design question for the full run.

Gate criteria (tracker decision rows dated 2026-11-28; thresholds 0.8 / 0.5):
  PRIMARY  Spearman of per-clip perclos80_mean, synthetic under the own-clip policy (A) vs the
           driving ground truth.
  ADDED    the same Spearman under the PRODUCTION policy (P): synthetic baseline fixed per
           identity, pooled p2/p90 over all 8 of that identity's clips; driving side the
           subject's alert-video baseline (= the ground truth). Declared before any Gate 4
           correlation existed.
  Scale-up needs BOTH >= 0.8, or both in [0.5, 0.8) with a tuning round in between.

Aggregations:
  A  synthetic, own-clip baseline            (primary)
  P  synthetic, per-identity pooled baseline (added criterion; derived manifest with
                                              subject_id = identity and class blanked, so
                                              build_baselines pools all 8 clips)
  C  driving crops, own-clip baseline        (control: same policy, crop and re-encode on real
                                              faces, so A vs C isolates reenactment itself)
A clip whose own-clip baseline Phase 1 refuses (degenerate EAR span: the eye barely moves) is
kept as a row marked `unmeasurable`, never dropped, and counted in the report.

Frame-level measures, independent of any baseline policy (synthetic frame i is driven by
driving frame i): `ear_frame_corr`, the Pearson correlation of the two EAR series, and
`closure_depth_ratio`, the synthetic EAR span (p90 - p2) over the driving one; below 1 means
the synthetic eye closes less deeply than the driver's.

POST-HOC DIAGNOSTICS (requested 2026-11-28 after 5 clips' raw EAR traces were inspected, NOT
pre-registered; the gate is the two PERCLOS Spearmans above). They explain the result, not decide it:
  D1 mean openness, synthetic vs driving (threshold-free)
  D2 openness range p90 - p2, synthetic vs driving (amplitude transfer)
  D3 compression factor = synthetic range / driving range, its spread, and whether it varies
     with driving PERCLOS (constant -> correctable by normalisation; variable -> not)
  D4 PERCLOS80 with each synthetic clip rescaled to its own p2/p90, computed independently
     from the frame tables. Phase 1's baseline IS p2/p90 (config.EAR_CLOSED_PERCENTILE,
     EAR_OPEN_PERCENTILE), so D4 must reproduce A: a cross-check, not a new estimate.
Scales: driving openness = original source frames of the cut, subject alert-video baseline;
synthetic openness = the production per-identity pooled baseline (fixed across that identity's
8 clips), because a
per-clip scale would cancel the very amplitude D2/D3 measure. Raw-EAR ratios alongside.

Ground truth is the driving interval's perclos80_cut (subject's alert-video baseline, exact
frames cut), with the original window value perclos80_mean alongside.

Runs in the `drowsy` env:  python -m gen.pilot_gate4
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase1_baseline"))
from src import aggregate as p1agg, blinks as p1blinks  # noqa: E402  unchanged Phase 1 code

HERE = Path(__file__).resolve().parents[1]
PILOT = HERE / "pilot"
P1 = HERE.parent / "phase1_baseline"
RES = PILOT / "results"
FRAMES = PILOT / "frames"
METRICS = ("perclos80_mean", "blink_rate_per_min", "mean_reopening_speed", "detection_rate")
# Reference categorical palette slots 1-4 (dataviz skill), plus a marker shape each so identity
# never rests on colour alone.
STYLE = {"syn01": ("#257de0", "o"), "syn02": ("#f47230", "s"),
         "syn03": ("#00b894", "^"), "syn04": ("#f2a600", "D")}


def run_p1(args: list[str], log: Path) -> None:
    r = subprocess.run([sys.executable, "-m", *args], cwd=P1, capture_output=True, text=True)
    log.write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"phase1 {' '.join(args[:1])} failed, see {log}")


def pooled_identity(manifest: Path, out: Path) -> Path:
    """subject_id = identity, class blank: phase1 build_baselines finds no alert row and pools
    every clip of the identity (recorded as baseline_source fallback:any-class)."""
    m = pd.read_csv(manifest, dtype=str)
    m["subject_id"] = m["identity_id"]
    m["class"] = ""
    m.to_csv(out, index=False)
    return out


def own_clip(manifest: Path, out: Path) -> Path:
    m = pd.read_csv(manifest, dtype=str)
    m["subject_id"] = m["video_id"]
    m.to_csv(out, index=False)
    return out


def aggregate(manifest: Path, name: str) -> pd.DataFrame:
    out = RES / f"features_{name}.csv"
    run_p1(["src.aggregate", "--manifest", str(manifest), "--frames", str(FRAMES),
            "--out", str(out), "--events-out", str(RES / f"events_{name}.csv")],
           RES / f"aggregate_{name}.log")
    return pd.read_csv(out, dtype={"video_id": str, "subject_id": str})


def corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    rho, prho = spearmanr(x[ok], y[ok])
    r, pr = pearsonr(x[ok], y[ok])
    return {"n": int(ok.sum()), "spearman_rho": rho, "spearman_p": prho,
            "pearson_r": r, "pearson_p": pr}


def cluster_boot_ci(df: pd.DataFrame, xcol: str, ycol: str, n_boot=5000, seed=0) -> tuple:
    """Spearman CI resampling DRIVING clips (each carries its 4 identities), since the 32
    points are 8 driving values seen 4 times, not 32 independent ones."""
    rng = np.random.default_rng(seed)
    df = df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])]
    clips = df.driving_clip_id.unique()
    groups = {c: df[df.driving_clip_id == c] for c in clips}
    vals = []
    for _ in range(n_boot):
        draw = pd.concat([groups[c] for c in rng.choice(clips, len(clips))])
        if draw[xcol].nunique() < 3:
            continue
        vals.append(spearmanr(draw[xcol], draw[ycol])[0])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def ear_compare(syn_id: str, drv_id: str) -> dict:
    a = pd.read_parquet(FRAMES / f"{drv_id}.parquet").ear_corrected.to_numpy(float)
    b = pd.read_parquet(FRAMES / f"{syn_id}.parquet").ear_corrected.to_numpy(float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    span = lambda e: float(np.nanpercentile(e, 90) - np.nanpercentile(e, 2))
    return {"ear_frame_corr": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
            "ear_span_driving": span(a), "ear_span_synthetic": span(b),
            "closure_depth_ratio": span(b) / span(a)}


def diagnostics(df: pd.DataFrame, syn_manifest: Path) -> pd.DataFrame:
    """Per-clip D1-D4 inputs. See module docstring."""
    p1man = pd.read_csv(P1 / "data" / "manifest.csv", dtype=str)
    drv = pd.read_csv(PILOT / "driving" / "driving_clips.csv", dtype={"subject_id": str})
    sm = pd.read_csv(syn_manifest, dtype=str)
    pooled = sm.assign(subject_id=sm.identity_id, **{"class": ""})
    id_base, problems = p1agg.build_baselines(pooled, FRAMES, use_corrected=True)
    if problems:
        print("identity baselines:", problems)
    rng = lambda o: float(np.nanpercentile(o, 90) - np.nanpercentile(o, 2))
    rows = []
    for r in df.itertuples():
        d = drv[drv.driving_clip_id == r.driving_clip_id].iloc[0]
        src = p1agg.load_frames(P1 / "data" / "frames", d.video_id)
        sbase, _ = p1agg.build_baselines(p1man[p1man.subject_id == d.subject_id],
                                         P1 / "data" / "frames", use_corrected=True)
        seg = src.iloc[int(d.start_frame):int(d.start_frame) + int(d.n_frames)]
        o_d = p1blinks.openness(p1agg.ear_column(seg, True), sbase[d.subject_id])
        syn = p1agg.load_frames(FRAMES, r.video_id)
        ear_s = p1agg.ear_column(syn, True)
        o_s = p1blinks.openness(ear_s, id_base[r.identity_id])
        own = p1blinks.estimate_baseline(ear_s, source="own-clip p2/p90")
        if own.span > 0:
            summ, _ = p1agg.summarise_frames(syn.reset_index(drop=True), own,
                                             p1agg.video_fps(syn), use_corrected=True)
            d4 = float(summ["perclos80_mean"])
        else:
            d4 = float("nan")
        rows.append({"video_id": r.video_id,
                     "d1_mean_open_driving": float(np.nanmean(o_d)),
                     "d1_mean_open_synthetic": float(np.nanmean(o_s)),
                     "d2_range_driving": rng(o_d), "d2_range_synthetic": rng(o_s),
                     "d3_compression": rng(o_s) / rng(o_d),
                     "d4_perclos_own_p2p90": d4,
                     "own_span_ear": own.span})
    return pd.DataFrame(rows)


def frame_checks(path: Path, n_expected: int) -> dict:
    """Automatic failure flags: frame count, frozen output, all-black frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n, diffs, dark, prev = 0, [], 0, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (128, 128)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is not None:
            diffs.append(np.abs(g - prev).mean())
        dark += int(g.mean() < 10)
        prev, n = g, n + 1
    cap.release()
    return {"frames_read": n, "frames_expected": n_expected,
            "median_frame_diff": float(np.median(diffs)) if diffs else float("nan"),
            "frozen_fraction": float(np.mean(np.array(diffs) < 0.05)) if diffs else float("nan"),
            "black_frames": dark}


def contact_sheet(clips: pd.DataFrame, out: Path) -> None:
    import cv2
    rows = []
    for _, r in clips.iterrows():
        cap = cv2.VideoCapture(str(PILOT / r.relpath))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles = []
        for k in np.linspace(0, n - 1, 6).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, f = cap.read()
            f = cv2.resize(f, (160, 160)) if ok else np.zeros((160, 160, 3), np.uint8)
            tiles.append(f)
        cap.release()
        row = np.hstack(tiles)
        cv2.putText(row, r.video_id.replace("p2pilot-lp-", ""), (4, 14), 0, 0.45, (0, 0, 255), 1)
        rows.append(row)
    cv2.imwrite(str(out), np.vstack(rows))


def scatter(df: pd.DataFrame, out: Path, ycol: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    lim = max(df.perclos80_cut.max(), df[ycol].max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#9c9c9c", lw=1, ls="--", zorder=1, label="y = x")
    for ident, g in df.groupby("identity_id"):
        c, m = STYLE[ident]
        ax.scatter(g.perclos80_cut, g[ycol], s=46, color=c, marker=m, edgecolor="white",
                   linewidth=1.2, zorder=3, label=f"{ident} ({g.collapsed_band.iloc[0]}, "
                                                   f"{g.gender_presentation.iloc[0][0].upper()})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Driving clip PERCLOS80 (ground truth, subject alert baseline)")
    ax.set_ylabel("Synthetic clip PERCLOS80")
    ax.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gen.pilot_manifest"], cwd=HERE, check=True)
    syn_m, drv_m = PILOT / "synthetic_manifest.csv", PILOT / "driving_manifest.csv"
    for m, name in ((syn_m, "synthetic"), (drv_m, "driving")):
        run_p1(["src.extract", "--manifest", str(m), "--root", str(PILOT),
                "--out", str(FRAMES)], RES / f"extract_{name}.log")

    A = aggregate(own_clip(syn_m, RES / "manifest_synthetic_ownclip.csv"), "A_synthetic_ownclip")
    P = aggregate(pooled_identity(syn_m, RES / "manifest_synthetic_pooled.csv"),
                  "P_synthetic_production")
    C = aggregate(own_clip(drv_m, RES / "manifest_driving_ownclip.csv"), "C_driving_ownclip")

    sm = pd.read_csv(syn_m, dtype=str)
    drv = pd.read_csv(PILOT / "driving" / "driving_clips.csv", dtype={"subject_id": str})
    ids = pd.read_csv(PILOT / "identities" / "chosen.csv")
    df = (sm[["video_id", "identity_id", "driving_clip_id", "driving_subject_id",
              "driving_fold", "driving_class", "relpath", "provenance_id"]]
          .merge(A[["video_id", *METRICS, "baseline_source"]], on="video_id", how="left")
          .merge(P[["video_id", "perclos80_mean", "baseline_source"]]
                 .rename(columns={"perclos80_mean": "perclos80_P", "baseline_source": "baseline_source_P"}),
                 on="video_id", how="left")
          .merge(C[["video_id", *METRICS]].rename(columns={"video_id": "driving_clip_id",
                 **{k: f"{k}_C" for k in METRICS}}), on="driving_clip_id")
          .merge(drv[["driving_clip_id", "perclos80_cut", "perclos80_mean", "nudge_tier",
                      "start_openness", "blink_rate_cut", "mean_reopening_speed_cut",
                      "detection_rate_cut", "n_frames"]]
                 .rename(columns={"perclos80_mean": "perclos80_window"}), on="driving_clip_id")
          .merge(ids[["identity_id", "collapsed_band", "gender_presentation", "ita_deg"]],
                 on="identity_id"))
    if len(df) != 32:
        raise RuntimeError(f"expected 32 joined clips, got {len(df)}")
    df["unmeasurable"] = df["baseline_source"].isna()
    df["unmeasurable_reason"] = np.where(df.unmeasurable, "own-clip eye baseline degenerate "
                                         "(phase1 aggregate refused it)", "")
    checks = pd.DataFrame([{"video_id": r.video_id,
                            **frame_checks(PILOT / r.relpath, r.n_frames),
                            **ear_compare(r.video_id, r.driving_clip_id)}
                           for r in df.itertuples()])
    df = df.merge(checks, on="video_id")
    df = df.merge(diagnostics(df, syn_m), on="video_id")
    df.to_csv(RES / "gate4_clips.csv", index=False)

    rows = []
    def add(comparison, sub, x, y, scope):
        rows.append({"comparison": comparison, "scope": scope, **corr(sub[x], sub[y])})
    for scope, sub in [("all 32", df)] + [(i, g) for i, g in df.groupby("identity_id")]:
        add("PRIMARY A synthetic own-clip vs driving ground truth", sub, "perclos80_cut", "perclos80_mean", scope)
    for scope, sub in [("all 32", df)] + [(i, g) for i, g in df.groupby("identity_id")]:
        add("ADDED CRITERION P synthetic production baseline vs driving ground truth", sub,
            "perclos80_cut", "perclos80_P", scope)
    add("A synthetic own-clip vs C driving own-clip", df, "perclos80_mean_C", "perclos80_mean", "all 32")
    c8 = df.drop_duplicates("driving_clip_id")
    add("C driving own-clip vs driving ground truth (policy + crop cost)", c8, "perclos80_cut", "perclos80_mean_C", "8 driving clips")
    mean8 = df.groupby("driving_clip_id").agg(gt=("perclos80_cut", "first"), syn=("perclos80_mean", "mean"))
    add("A mean over 4 identities vs driving ground truth", mean8, "gt", "syn", "8 driving clips")
    add("frame EAR: synthetic vs driving EAR span (closure depth)", df, "ear_span_driving",
        "ear_span_synthetic", "all 32")
    for x, y, name in [("blink_rate_cut", "blink_rate_per_min", "blink rate: synthetic A vs driving"),
                       ("mean_reopening_speed_cut", "mean_reopening_speed", "reopening speed: synthetic A vs driving")]:
        add(name, df, x, y, "all 32")
    for scope, sub in [("all 32", df)] + [(i, g) for i, g in df.groupby("identity_id")]:
        add("POST-HOC D1 mean openness: synthetic (identity baseline) vs driving", sub,
            "d1_mean_open_driving", "d1_mean_open_synthetic", scope)
        add("POST-HOC D2 openness range p90-p2: synthetic vs driving", sub,
            "d2_range_driving", "d2_range_synthetic", scope)
        add("POST-HOC D4 PERCLOS80 own p2/p90 rescale vs driving ground truth", sub,
            "perclos80_cut", "d4_perclos_own_p2p90", scope)
        add("POST-HOC D3 compression factor vs driving PERCLOS (is it constant?)", sub,
            "perclos80_cut", "d3_compression", scope)
    summary = pd.DataFrame(rows)
    comp = df.groupby("identity_id").d3_compression.agg(["mean", "std", "min", "max"])
    comp.loc["all 32"] = [df.d3_compression.mean(), df.d3_compression.std(),
                          df.d3_compression.min(), df.d3_compression.max()]
    comp["cv"] = comp["std"] / comp["mean"]
    comp.to_csv(RES / "gate4_compression.csv")
    print("D3 compression factor (synthetic range / driving range):")
    print(comp.round(3).to_string())
    d4_vs_a = (df.d4_perclos_own_p2p90 - df.perclos80_mean).abs().max()
    print(f"D4 cross-check: max |D4 - A| = {d4_vs_a:.4f} (should be ~0)")
    for comp_name, col in (("PRIMARY A synthetic own-clip vs driving ground truth", "perclos80_mean"),
                           ("ADDED CRITERION P synthetic production baseline vs driving ground truth", "perclos80_P")):
        lo, hi = cluster_boot_ci(df, "perclos80_cut", col)
        k = summary[(summary.comparison == comp_name) & (summary.scope == "all 32")].index[0]
        summary.loc[k, "spearman_ci_lo"], summary.loc[k, "spearman_ci_hi"] = lo, hi
    summary.to_csv(RES / "gate4_summary.csv", index=False)

    per_id = df.groupby("identity_id").agg(
        band=("collapsed_band", "first"), gender=("gender_presentation", "first"),
        ita=("ita_deg", "first"), detection_mean=("detection_rate", "mean"),
        detection_min=("detection_rate", "min"), perclos_mean=("perclos80_mean", "mean"),
        blink_rate=("blink_rate_per_min", "mean"), reopen=("mean_reopening_speed", "mean"),
        ear_frame_corr=("ear_frame_corr", "median"),
        closure_depth_ratio=("closure_depth_ratio", "median"),
        unmeasurable=("unmeasurable", "sum"))
    per_id.to_csv(RES / "gate4_per_identity.csv")

    scatter(df, RES / "gate4_scatter.png", "perclos80_mean",
            "Gate 4: synthetic vs driving PERCLOS80 (synthetic: own-clip baseline)")
    scatter(df, RES / "gate4_scatter_production.png", "perclos80_P",
            "Added criterion: synthetic PERCLOS80, production (per-identity pooled) baseline")
    contact_sheet(df.sort_values(["identity_id", "perclos80_cut"]), RES / "gate4_contact_sheet.jpg")

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(per_id.round(3).to_string())
    print(df[["video_id", "perclos80_cut", "perclos80_mean", "perclos80_mean_C", "perclos80_P",
              "ear_frame_corr", "closure_depth_ratio", "unmeasurable", "detection_rate", "frames_read", "frames_expected", "frozen_fraction",
              "black_frames"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
