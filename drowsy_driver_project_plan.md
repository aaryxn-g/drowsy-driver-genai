# Synthetic Drowsy-Driver Data Generation & Fatigue Detection
## Full Project Plan of Action

---

## 0. Project framing

**Core research question:** Can generator-produced (diffusion/GAN reenactment) synthetic drowsiness footage, injected into training in a controlled ratio, close the demographic accuracy gap of a fatigue detector without degrading overall F1, inflating false-alarm rate, or requiring new real driver footage?

**Two deliverables:**
1. A fatigue detector (PERCLOS/yawn/head-pose based) with documented, quantified fairness improvement.
2. A shareable synthetic drowsiness dataset (with clear synthetic-data provenance tags — never mixed anonymously with real data downstream).

**Governing principle for the whole plan:** every design choice (which architecture, which synthetic ratio, which augmentation strategy) must come from a **head-to-head benchmark you run yourself**, not from a single paper's claim. The plan is structured so each phase ends with a comparison table/chart that becomes the justification for the next phase's choice.

---

## Phase 0 — Scoping, literature grounding, and resource audit (Week 1–2)

**Goal:** Fix the exact problem definition before touching code, and confirm you actually *can* get the data and compute.

- **0.1 Systematic mini-review.** Don't just cite the 3 datasets in the brief — pull the last 12–18 months of literature on (a) drowsiness/fatigue detection architectures, (b) face reenactment/diffusion editing, (c) synthetic-data-for-fairness in face-based ML. Recent evidence to anchor on:
  - A 2025 PRISMA systematic review (33 studies, 2021–2025) found lightweight CNN hybrids like **YOLOv8-FD (98.06% @ 16.7ms)** and **EffRes-DrowsyNet (98.07% @ 28ms)** dominate the accuracy–latency Pareto frontier, while **transformer + diffusion-denoising architectures exceeded real-time latency budgets by 1–2 orders of magnitude** — a critical constraint for you, since your detector must stay deployable even after retraining on synthetic-augmented data.
  - A 2025 Nature Sci. Reports paper reports **ViT (99.15%) and Swin Transformer (99.03%)** beating CNN baselines like VGG19 (98.7%) on eye-state classification (MRL Eye Dataset), arguing transformers capture long-range facial cues (brow furrowing, slow blinks) that localized CNN receptive fields miss — relevant to your architecture shortlist.
  - On the generative side, 2025 work (FRVD — Face Reenactment Video Diffusion) shows video-diffusion-based reenactment now handles **large head-pose variation** far better than GAN/keypoint-warping methods, which is exactly your head-nodding cue — a strong candidate over older first-order-motion-model GANs.
  - On synthetic-data-for-fairness, results are **mixed and ratio-dependent**, not a guaranteed win: a 2024 face-recognition fairness study found synthetic-only training lags real data by a wide margin and that **naive data augmentation can worsen fairness even as raw accuracy ticks up**; balanced real+synthetic mixes reduced demographic accuracy gaps, but pure synthetic never matched real. This directly justifies your "chart accuracy and subgroup-gap vs. synthetic ratio" step — the literature says the optimum is not "more synthetic = better," it's an empirical sweet spot you must find.
  - Also flag for your writeup: deepfake/synthetic-face literature explicitly warns that detectors **can overfit to generator-specific artifacts**, and fairness gains from synthetic augmentation **may fail to transfer to real footage** — this is precisely why you need the Phase 5 stress test on unseen real conditions, not just synthetic-heldout numbers.
- **0.2 Dataset access audit.** NTHU-DDD and UTA-RLDD both require signed academic-use agreements from their host labs/universities — start these requests immediately (Week 1), since approval can take 1–3 weeks and blocks Phase 1. Confirm FFHQ/CelebA licensing terms (research-only, no re-identification) since your generator will condition on these identities.
- **0.3 Compute audit.** Diffusion-based video reenactment training/inference is GPU-heavy. Decide now: are you *training* a reenactment model from scratch (needs serious GPU budget) or *fine-tuning/using a pretrained* open-source reenactment/diffusion model (far more realistic for a course/thesis timeline)? Recommendation below assumes the latter.
- **0.4 Ethics/IRB check.** Even with FFHQ/CelebA (public, consented-at-collection-time identities) and NTHU-DDD/UTA-RLDD (existing consented datasets), confirm your institution's IRB/ethics board sign-off for face data + generative identity synthesis, and plan the dataset release (Output b) datasheet now (see Phase 6.5) rather than retrofitting it later.

**Phase 0 deliverable:** a scoping document — problem statement, confirmed dataset access status, compute plan, and an annotated bibliography table (method / year / reported metric / relevance) for each of the three literature strands above.

---

## Phase 1 — Baseline fatigue detector + subgroup audit (Week 3–5)

**Goal:** Establish the pre-intervention numbers everything else is measured against. This is the single most important phase to get right — a sloppy baseline invalidates every later chart.

### 1.1 Data preparation
- Pool NTHU-DDD + UTA-RLDD. Extract per-frame/per-clip labels: eye state (open/closed), yawn state, head-pose angle (pitch/yaw/roll via a landmark model, e.g. MediaPipe FaceMesh or 6DRepNet).
- **Annotate demographic metadata** for every subject: skin tone (Fitzpatrick or ITA-based, not self-declared race — self-declared race is not directly derivable from video and is ethically fraught; use a validated skin-reflectance proxy instead), apparent age bracket, gender presentation, glasses/no-glasses. This subgroup table is what every later "gap" number is computed against — get an independent second annotator to cross-check a sample and report inter-annotator agreement (Cohen's κ).
- Report class balance and subgroup balance explicitly (a table). This *is* your scarcity/imbalance evidence — the headline justification for the whole project — so make it rigorous, not anecdotal.

### 1.2 Architecture shortlist (empirically compared, not assumed)
Train and directly compare on your pooled real dataset:
| Candidate | Why in the shortlist | Expected trade-off |
|---|---|---|
| EfficientNet/ResNet-hybrid CNN (e.g., EffRes-style) | Reported ~98% accuracy at ~28ms latency in 2025 review | Best accuracy/latency balance |
| YOLOv8-based face-region + state classifier | ~98% @ 12–17ms, fastest | Best for edge/real-time deployment framing |
| ViT / Swin Transformer | 99%+ reported on eye-state, better global context | Higher latency, needs more data to avoid overfitting |
| Lightweight LSTM/temporal-CNN on PERCLOS time series | Captures temporal dynamics (blink duration, nod rate) directly | Needs windowed sequences, not single frames |

Train each on identical splits, report accuracy/F1/latency, and **pick the baseline architecture from this table with numeric justification**, not from the brief's suggestion alone — if a CNN hybrid clearly wins accuracy-per-millisecond, say so and use it; if the transformer's fairness/robustness profile turns out better in 1.3, that's a legitimate reason to trade some latency for it. Document the decision either way.

### 1.3 Baseline subgroup accuracy audit
- Compute overall F1 and **per-subgroup F1** (skin tone × gender-presentation × glasses) for the chosen architecture.
- Report the **gap metrics** you'll track for the rest of the project: max-min subgroup F1 gap, standard deviation across subgroups, and equalized-odds gap (false-alarm-rate difference across subgroups — this anticipates Phase 6.3).
- This gap number is your headline "before" figure. Bootstrap confidence intervals (not just point estimates) so later improvements can be tested for statistical significance, not eyeballed off a bar chart.

**Phase 1 deliverable:** chosen baseline architecture + full report (overall F1, per-subgroup F1 with CIs, latency, dataset balance table).

---

## Phase 2 — Build the generator (Week 5–9)

**Goal:** A conditional pipeline: identity image (FFHQ/CelebA) + drowsiness-cue parameters (from a real driver template) → synthetic video clip exhibiting that cue on that identity, under varied lighting/pose/camera angle.

### 2.1 Technique shortlist (benchmarked, not assumed)
Given the literature above, GAN-based first-order-motion-style reenactment is the *older* default; video-diffusion reenactment is the *current* SOTA for exactly your hardest case (head-nodding = large pose change). Recommended comparison set:

| Approach | Class | Strength for this task | Risk |
|---|---|---|---|
| First-Order-Motion-Model / keypoint-warping GAN (e.g., FOMM, Face vid2vid style) | GAN, keypoint-driven | Fast, lightweight, well-understood, easy to fine-tune | Warping artifacts under large pose (exactly your head-nod cue) |
| Video-diffusion reenactment (FRVD-style: motion extractor + warping-feature-mapper into a pretrained I2V diffusion latent space) | Diffusion | State of the art for large-pose fidelity per 2025 results; strong identity preservation | Heavier compute, slower inference, needs a decent pretrained I2V backbone |
| Audio/expression-conditioned diffusion talking-head models (e.g., EMO-style, motion-prior conditional diffusion) | Diffusion | Good for yawn/mouth dynamics specifically | Built for speech-driven lip sync, needs repurposing for yawn-cue conditioning rather than audio |
| Attribute-editing diffusion (facial-attribute edit, not full reenactment) | Diffusion, image-space | Simpler for eye-closure (a more local, less pose-heavy edit) | Weaker for full head-nod motion transfer |

**Recommendation to actually implement, pending your own pilot test:** use a **hybrid** — GAN/keypoint-warping (fast) as your first working pipeline to unblock Phase 3 quickly, then run a **head-to-head pilot** (50 clips) against a diffusion-reenactment pipeline on identity-preservation, motion fidelity, and visual quality metrics (below). Whichever wins on the metrics that matter for your downstream detector (motion fidelity for head-nod, not just FID) becomes the pipeline you scale up. This mirrors exactly the kind of "backed by data and numbers" decision you asked for, and avoids over-committing to the heavier diffusion pipeline if the GAN pipeline is good enough for your detector's actual sensitivity.

### 2.2 Cue-parameter design
Define a small, explicit parameter set extracted from real drowsiness templates in NTHU-DDD/UTA-RLDD:
- Eye closure: PERCLOS value/trajectory, blink duration
- Yawn: mouth-aspect-ratio trajectory, duration
- Head nod: pitch trajectory, nod frequency/amplitude
Build a **template library**: for each drowsiness severity level (alert/low/medium/high, matching the fuzzy-logic 3-level scheme already used in the literature), extract a set of real cue trajectories to drive the generator, decoupled from the source identity.

### 2.3 Diversity injection
Vary systematically and log every combination: identity (FFHQ/CelebA sampled to intentionally **over-represent currently under-represented subgroups from your Phase 1.3 audit** — this is the fairness-targeting mechanism, not random sampling), lighting (day/dusk/simulated IR-night), camera angle (dashboard-mount range), and occlusion (glasses on/off, partial face crop).

### 2.4 Generator quality evaluation (before it ever touches the detector)
Don't skip straight to using the generator — validate it first:
- **Identity preservation:** face-embedding cosine similarity (e.g., ArcFace) between source identity and generated frames.
- **Motion/cue fidelity:** re-run your Phase-1 PERCLOS/yawn/head-pose extractors on the *generated* video and compare against the intended template parameters (this checks the generator actually produced the drowsiness cue you asked for, not just a plausible face).
- **Visual quality/realism:** FID/FVD against real footage, plus a small human perceptual study (e.g., 20–30 raters, real-vs-synthetic forced choice) if time allows.
- **Failure-mode logging:** systematically log where warping artifacts, temporal flicker, or identity drift occur (expected to correlate with large head-pose, which is your literature-flagged risk area) — this becomes a documented limitation, not a surprise later.

**Phase 2 deliverable:** chosen generator pipeline (justified by the pilot comparison), cue-template library, quality-validated synthetic clip generation pipeline, and a quality report (identity/motion-fidelity/FID numbers).

---

## Phase 3 — Synthetic dataset construction (Week 9–11)

**Goal:** Produce the actual shareable synthetic dataset at scale, with proper structure for the ratio sweep in Phase 4.

- Generate synthetic clips at multiple **target subgroup-balancing levels** — e.g., a version that mirrors the real dataset's (imbalanced) subgroup distribution, and a version that's explicitly balanced/inverse-weighted to fill the gaps found in Phase 1.3. This lets Phase 4 separate two different questions: "does adding *any* synthetic data help?" vs. "does *targeted* synthetic data (aimed at underrepresented subgroups) help more than untargeted synthetic data?" — the second is the actually novel contribution.
- Tag every synthetic sample with full provenance metadata (source identity ID, cue template, subgroup label, generation parameters) — required both for reproducibility and for the dataset-release deliverable.
- Run an automated QC pass (reuse Phase 2.4 metrics) to filter out low-fidelity generations before they enter training — report the filtered-out rate as a data point (it tells you the generator's real yield/cost).

**Phase 3 deliverable:** structured synthetic dataset (raw + balanced variants), QC report, dataset card draft.

---

## Phase 4 — Retraining sweep: real+synthetic mixes (Week 11–14)

**Goal:** The central empirical result — accuracy and subgroup-gap as a function of synthetic ratio, done properly (not a single mix).

- **4.1 Sweep design.** Train the Phase-1 architecture at multiple real:synthetic ratios (e.g., 100:0, 90:10, 75:25, 50:50, 25:75, and a synthetic-only 0:100 control), each repeated with **multiple random seeds** (minimum 3) to get variance estimates — a single run per ratio is not sufficient to claim a trend, especially given the literature's warning that synthetic augmentation effects on fairness can be small and noisy.
- **4.2 Two sweep tracks**, per the Phase 3 note: (a) untargeted synthetic mix, (b) subgroup-targeted synthetic mix. Compare both against the real-only baseline **and against each other** — the targeted-vs-untargeted comparison is the strongest evidence for your "targeted synthetic data" fairness claim.
- **4.3 Metrics per ratio:** overall F1, per-subgroup F1, max-min gap, equalized-odds gap, plus a **held-out real-only test set that never appears in any training mix** (critical — if the same real subjects contribute to both train and eval, your fairness numbers are contaminated).
- **4.4 Output chart(s):** overall accuracy vs. synthetic ratio, and subgroup gap vs. synthetic ratio, both tracks overlaid, with error bars from the seed repeats. This is the chart your brief explicitly asks for — treat it as the centerpiece figure of the whole project.
- **4.5 Pick the operating point.** From this chart, select the synthetic ratio that best trades off overall F1 loss (if any) against subgroup-gap reduction, and justify it numerically (e.g., "25% targeted-synthetic reduces subgroup gap by X points at a cost of Y points overall F1, which we judge acceptable because...").

**Phase 4 deliverable:** the ratio-sweep chart(s), the selected operating-point model, and a written justification of the trade-off chosen.

---

## Phase 5 — Stress-testing under unseen conditions (Week 14–16)

**Goal:** Check whether the fairness gains from Phase 4 are real or an artifact of testing on synthetic-adjacent distributions — directly addressing the "overfits to generator artifacts" risk flagged in the literature review.

- Evaluate the Phase-4 operating-point model (and the real-only baseline, for comparison) on:
  - **Night/IR-style imagery** — either real IR clips if available in your datasets, or a held-out simulated-IR subset.
  - **Glasses/occlusion** — held-out real clips with glasses, sunglasses, partial occlusion (hand near face, phone, mask).
  - **Cross-dataset generalization** — train on NTHU-DDD-heavy mix, test on UTA-RLDD-only (or vice versa), to check the synthetic augmentation isn't just overfitting to one dataset's camera/lighting signature.
- Report the **same subgroup-gap metrics** under each stress condition, not just overall accuracy — the fairness claim has to survive out-of-distribution testing to be credible, and this is exactly where the deepfake-detection literature above warns synthetic-driven fairness gains most often collapse.
- If gains collapse under stress-test conditions, that's a legitimate and reportable finding — document it honestly rather than only reporting the favorable Phase 4 numbers.

**Phase 5 deliverable:** stress-test report comparing baseline vs. synthetic-augmented model across night/IR, occlusion, and cross-dataset conditions.

---

## Phase 6 — Responsible-AI layer (runs alongside Phases 4–5, consolidated Week 16–17)

- **6.1 Fairness (headline).** Consolidate the subgroup-gap results from Phases 1, 4, 5 into a single before/after fairness narrative with confidence intervals and stress-test caveats.
- **6.2 Privacy.** Document precisely what real data was used (only existing consented datasets, no new driver recruitment), and run a **membership-inference / identity-leakage check** on the generator — i.e., verify synthetic identities aren't just near-duplicates of specific FFHQ/CelebA source images that could re-identify someone. This is a known failure mode in synthetic face generation (flagged in the biometric-fairness literature above) and should be tested, not assumed away.
- **6.3 False-alarm analysis.** Report false-positive drowsiness-flag rate **per subgroup**, separately from overall F1 — a model can have equal F1 across groups while still over-flagging one group more than another, which is the concrete "disciplinary consequence" harm the brief calls out. This uses the equalized-odds gap metric already computed in 1.3/4.3/5.
- **6.4 Limitations section.** Explicitly state where synthetic data did *not* help (e.g., if stress-test gains collapsed), and why (generator artifact overfitting, domain gap) — citing the literature's own documented version of this failure mode strengthens rather than weakens the report.
- **6.5 Dataset card / datasheet.** Prepare a formal datasheet-for-datasets style document for the synthetic dataset deliverable (composition, collection process, intended/inappropriate uses, provenance tags from Phase 3, known limitations) before release.

---

## Phase 7 — Consolidation, validation, and writeup (Week 17–19)

- **7.1 Ablations** (time permitting, prioritized): generator choice (GAN vs diffusion pipeline) effect on downstream fairness; targeted vs. untargeted synthetic sampling; architecture choice (CNN-hybrid vs transformer) effect on both latency and fairness gap.
- **7.2 Reproducibility pass.** Fix seeds, freeze splits, package code/configs, re-run the final headline numbers once end-to-end to confirm they reproduce.
- **7.3 Final report structure:** baseline + gap (Phase 1) → generator validation (Phase 2) → dataset (Phase 3) → ratio-sweep chart (Phase 4, the centerpiece) → stress test (Phase 5) → responsible-AI section (Phase 6) → limitations → dataset release.
- **7.4 Deliverable packaging:** fatigue detector (model + inference code), synthetic dataset + datasheet, all charts/tables, full report.

---

## Cross-cutting notes

- **Every "why did you pick X" question in a defense should map to a table or chart you generated in this plan** — architecture (1.2), generator (2.1 pilot), synthetic ratio (4.5), sampling strategy (4.2) — that's the "backed by data and numbers" requirement built into the structure, not bolted on at the end.
- **Timeline is ~19 weeks (a semester)** at a research-project pace; compress Phases 2–3 first if you need to shorten it, since the generator can start from a pretrained model rather than training from scratch — Phase 1 (baseline+audit) and Phase 4 (the sweep) are the two phases you should not compress, since they produce the core evidence.
- **Biggest early risk is dataset access latency** (NTHU-DDD/UTA-RLDD approval) — start those requests in Phase 0, Week 1, before anything else.
