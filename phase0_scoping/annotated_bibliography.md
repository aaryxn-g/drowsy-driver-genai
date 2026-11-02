# Annotated Bibliography — Phase 0.1

Format: **Claim / method** — Year — Source — Why it matters to this project

## A. Drowsiness/fatigue detection architectures

1. **PRISMA systematic review, 33 studies (2021–Oct 2025)** — 2025 — Frontiers in AI —
   Found lightweight CNN hybrids dominate the accuracy-latency Pareto frontier: ME-YOLOv8
   (≥97% @ 12.2ms), YOLOv8-FD (98.06% @ 16.7ms), EffRes-DrowsyNet (98.07% @ 28.0ms).
   Transformer + diffusion-denoising architectures exceeded a 33ms real-time threshold by
   1–2 orders of magnitude. **→ Directly informs Phase 1.2 architecture shortlist and the
   latency constraint on any architecture we retrain with synthetic data.**
   URL: frontiersin.org/.../frai.2026.1871926

2. **ViT/Swin Transformer for eye-state classification** — 2025 — Nature Sci. Reports
   (Hassan et al., DOI 10.1038/s41598-025-02111-x) — ViT 99.15%, Swin 99.03% vs VGG19 98.7%
   on MRL Eye Dataset; argues transformers capture long-range facial cues (brow furrowing,
   slow blinks) CNNs miss due to local receptive fields. **→ Counter-evidence to include in
   the architecture benchmark table — transformers may win on accuracy even if they lose on
   latency; this is exactly the trade-off Phase 1.2 needs numeric evidence for.**

3. **Survey: Intelligent driver monitoring systems** — 2024/2025 — Springer (Artificial
   Intelligence Review) — Broad modality survey (PERCLOS/ECD/AOT fusion, hybrid eye-mouth-
   head CNNs like EM-CNN). **→ Useful for Phase 1.1 cue-extraction pipeline design
   (PERCLOS/POM computation approach).**

*(Add more as you read — target 8-12 entries in this section before closing Phase 0.)*

## B. Face reenactment / diffusion-based generation

4. **FRVD — Face Reenactment Video Diffusion** — 2025 — Computers & Graphics /
   arXiv:2507.16341 — Motion extractor + Warping Feature Mapper projects warped source
   image into a pretrained image-to-video diffusion model's motion-aware latent space;
   explicitly targets the **large-pose problem** (warping artifacts under big head-pose
   change) that keypoint-warping GANs struggle with. **→ Head-nod is exactly a large-pose
   cue — this is the strongest diffusion-pipeline candidate for Phase 2.1.**

5. **FaceEditTalker: controllable talking-head generation with facial-attribute editing** —
   2025 — arXiv:2505.22141 — Combines attribute editing with talking-head generation,
   references EMO (audio2video diffusion), SkyReels-A1 (expressive portrait animation in
   video diffusion transformers). **→ Candidate for yawn/mouth-dynamics-specific cue
   conditioning in Phase 2.2.**

6. **Anchored Diffusion for Video Face Reenactment** — 2024 — arXiv:2407.15153 — Diffusion
   approach benchmarked against FID/FVD and VBench; useful reference for **Phase 2.4
   evaluation metric selection** (which FID/FVD variants are standard in this literature).

*(Add first-order-motion-model / GAN reenactment baselines here too, for the Phase 2.1 pilot's
"older" comparison arm — e.g., FOMM, face-vid2vid.)*

## C. Synthetic data for demographic fairness

7. **Bridging Racial and Age Gaps in Face Recognition: a Data-Augmentation Framework** —
   2025 — Springer SN Computer Science — Iterative scheme that always augments the
   *weakest* subgroup (active-curriculum style); compares standard augmentation vs
   StyleGAN-generated synthetic faces for closing Equalized-Odds gaps. **→ Direct template
   for Phase 3's "targeted" synthetic-balancing strategy — consider adopting their
   iterative/curriculum approach rather than a one-shot balanced batch.**

8. **The Impact of Balancing Real and Synthetic Data on Accuracy and Fairness in FR** —
   2024 — arXiv:2409.02867 — Key finding: synthetic-only training lags real data by ~15%
   verification accuracy; naive augmentation can *worsen* fairness even while raising
   accuracy; balanced real+synthetic mixes reduced gaps, but the win is not automatic.
   **→ This is the strongest caution in the whole bibliography — justifies why Phase 4 must
   sweep multiple ratios with multiple seeds rather than assuming "more synthetic = better."**

9. **GANDiff FR / "Face Recognition Accuracy Across Demographics"** — 2022–2023 —
   ResearchGate — Synthetic framework precisely controlling demographic/environmental
   factors; racial performance-gap reduced from 0.0107 to 0.0098 (std dev), overall accuracy
   also improved slightly. **→ Small-but-real effect size to calibrate expectations against —
   don't expect a huge gap closure from Phase 4; a modest, statistically significant
   reduction is a legitimate, publishable result.**

10. **Toward Calibrated, Fair, and Accurate Deepfake Detection** — 2025 — arXiv:2606.09881 —
    Explicitly documents that detectors trained on synthetic-augmented data **can overfit to
    generator-specific artifacts**, and fairness gains **may fail to transfer to real
    footage**. **→ Direct justification for Phase 5's stress-test design — this is the
    failure mode we are testing for.**

*(Add 2-3 more once you've done your own search pass — try "synthetic data domain gap
low-resolution face recognition 2026" and "membership inference synthetic face generation"
for Phase 6.2's privacy check methodology.)*

---

## How to keep extending this
Run these searches yourself (or ask me to) roughly once per phase, since this is a fast-
moving area:
- "driver drowsiness detection benchmark 2026"
- "video diffusion reenactment head pose 2026"
- "synthetic data fairness face recognition benchmark 2026"
- "membership inference attack generative face model"
- "datasheets for datasets template synthetic biometric"
