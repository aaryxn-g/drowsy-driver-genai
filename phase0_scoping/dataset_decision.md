# Final Dataset Decision (verified 27 Sep 2026)

Every source below was opened and checked live in a browser on 27 Sep 2026: official page, access
process, licence text and actual contents. Nothing here is taken from memory or a secondary summary.

---

## Final selection

| # | Dataset | Role in the project | Access | Licence | Status |
|---|---|---|---|---|---|
| 1 | **UTA-RLDD** | **Primary real dataset**: baseline training/eval, skin-tone fairness audit, Phase 4 test set | Direct Google Drive download, no form | Research use, cite CVPRW 2019 paper | Ready to download |
| 2 | **NTHU-DDD (official only)** | Drowsiness **cue templates** (acted yawn/nod/slow blink, per-frame labels); **night-IR + glasses/sunglasses stress test** (Phase 5) | Signed licence agreement (department head or lab director) + email | Non-commercial; **no redistribution**; face images may not be shown, except subjects 010 and 021 | Needs HoD signature |
| 3 | **YawDD** | Extra yawn-cue templates; **real in-car, colour** cross-domain test | IEEE DataPort, free IEEE account | Research use; screenshots only of participants who consented (see Participants Information file) | Ready to download |
| 4 | **FFHQ** | **Identity pool** for synthesis (high-res, 1024×1024) | GitHub download script, Google Drive | CC BY-NC-SA 4.0 (**so our synthetic dataset must also be released as CC BY-NC-SA 4.0**) | Ready to download (subset) |
| 5 | **FairFace** | (a) Pretrained race/gender/age classifier to **label FFHQ identities** so we can sample to fill gaps; (b) secondary identity pool with balanced demographics | GitHub links, Google Drive | CC BY 4.0 | Ready to download |

### Dropped, with reasons

| Dataset | Why dropped |
|---|---|
| **NTHU-DDD Kaggle mirrors** (`banudeep/nthuddd2`, `ikhlaselhamly/nthu-ddd`, `samymesbah/...`, and others) | `banudeep/nthuddd2` (the one cited in MDPI Sensors 2025) now returns "page not found". The live mirrors are **loose extracted frames only**: no videos, no subject IDs, licence "Unknown". Without subject IDs you can't make subject-level splits (so identities leak between train and test) or do demographic annotation. Without video you can't compute PERCLOS or extract nod/yawn motion. The NTHU licence also says the dataset "cannot be distributed to second parties", so these mirrors breach it. |
| **CelebA** | Demographics skew young, light-skinned and Western (celebrities), which is the opposite of what we are correcting. FairFace + FFHQ covers the same need better. |
| **DMD** | Official site says **IR and depth channels have been removed; RGB only**, so it no longer helps the night/IR stress test (my earlier suggestion to use it for that was wrong). About 25 TB raw, and the NoDerivatives licence. Optional at most. |
| **UL-DD (2025)** | Too new to have benchmark numbers to compare against; access not yet checked. Out of scope unless time allows. |

---

## Key facts that shape the methodology

### UTA-RLDD: this is where the imbalance lives (verified from the official page)
- 60 participants, 180 RGB videos (~10 min each, ~30 hrs), 3 classes: alert / low-vigilant / drowsy, based on the Karolinska Sleepiness Scale (KSS)
- **51 men vs 9 women** (15% women)
- Ethnicity: 10 Caucasian, 5 non-white Hispanic, 30 Indo-Aryan/Dravidian, 8 Middle Eastern, 7 East Asian. **No African-descent subjects are listed.**
- Age 20–59, mean 25 (SD 6), so very few older drivers
- Glasses in only **21 of 180** videos; heavy facial hair in 72/180
- Self-recorded on phone/webcam, frame rate below 30 fps, varied angles and backgrounds
- **5 official folds**: use one as test, four as train, and rotate, to stay comparable with published results
- **36 of 60** participants consented to face publication (spreadsheet on the official site). Only show those faces in figures.

These numbers **are the motivation section of the report**. The gaps we target with synthetic data:
**women, older drivers, darker skin tones (especially African descent, absent entirely), and glasses wearers.**

### NTHU-DDD: important limitation for the fairness audit
- 36 subjects, 5 scenarios (BareFace, Glasses, Sunglasses, Night_BareFace, Night_Glasses), ~9.5 hrs, 640×480 AVI
- Recorded with **active IR illumination, so the footage is effectively grayscale**
- **Consequence: ITA skin-tone measurement cannot be computed on NTHU-DDD** (ITA needs colour). Skin-tone subgroup analysis must use UTA-RLDD and YawDD (colour). On NTHU-DDD, only glasses/sunglasses and day/night subgroups can be analysed.
- Subjects look predominantly East Asian (collected in Taiwan), so there's little demographic variety
- Behaviour is **acted** (subjects instructed to yawn/nod), not genuine drowsiness. That's good for clean cue templates and weaker as a realistic test.
- **Per-frame drowsy/non-drowsy labels**, which make it the best source for extracting labelled cue segments
- Night scenarios are **15 fps**, the others 30 fps, so resample to a common rate
- Required citation: Weng, Lai, Lai, "Driver Drowsiness Detection via a Hierarchical Temporal Deep Belief Network", ACCV Workshops 2016

### YawDD
- 351 colour videos (640×480, 30 fps AVI) in a **real car**: 322 mirror-camera + 29 dash-camera; male/female, glasses/sunglasses, different ethnicities
- **Unlabelled**: yawn segments must be annotated by us. Plan: annotate the 29 dash-camera videos fully (each contains silent / talking / yawning), and annotate the mirror set only as needed.
- Check the "Allow picture to be shared publicly?" column before putting any frame in the report.

### FFHQ + FairFace pipeline (identity sampling)
1. Download FFHQ **thumbnails (1.95 GB)** plus the JSON metadata first. Don't pull the full 89 GB.
2. Run FairFace's pretrained classifier (`res34_fair_align_multi_7`) on the thumbnails to get race/gender/age estimates for all 70k.
3. Select identities to **fill the UTA-RLDD gaps** (women, 40+, darker skin tones, glasses), then download only those at 1024×1024 using the script.
4. Log these labels as *estimated* attributes in the provenance metadata. They come from a classifier, not self-report; say so in the dataset card.

---

## Action checklist

- [ ] **UTA-RLDD**: download all 10 zips (Fold1–5, part1/part2) from
      https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl
      Also download the "Image Publishable" consent spreadsheet from https://sites.google.com/view/utarldd/home
      Check total size in Drive (list view) before starting and make sure you have the disk space.
- [ ] **NTHU-DDD**: download the licence PDF:
      http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/NTHU-DDD-LicenseAgreement.pdf
      → fill in name/email/address/institution → **get it signed by your department head or lab director (with their title)**
      → email the signed form to **cvlablai636@my.nthu.edu.tw** with the subject exactly
      **"Dataset on Driver Drowsiness Detection from Video"**
- [ ] **YawDD**: create a free IEEE account → download `YawDD.rar.gz` (4.94 GB) + Participants Information from
      https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset
- [ ] **FFHQ**: `git clone https://github.com/NVlabs/ffhq-dataset` → `python download_ffhq.py --json --thumbs`
- [ ] **FairFace**: images (padding 0.25) + labels + pretrained model from https://github.com/joojs/fairface
      (code: https://github.com/dchen236/FairFace)

## Required citations (collect now for the report)
- UTA-RLDD: Ghoddoosian, Galib, Athitsos, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection", CVPR Workshops 2019 (arXiv:1904.07312)
- NTHU-DDD: Weng, Lai, Lai, ACCV Workshops 2016 (above)
- YawDD: Abtahi, Omidyeganeh, Shirmohammadi, Hariri, "YawDD: A Yawning Detection Dataset", ACM MMSys 2014, DOI 10.1145/2557642.2563678
- FFHQ: Karras, Laine, Aila, "A Style-Based Generator Architecture for GANs", CVPR 2019 (arXiv:1812.04948)
- FairFace: Kärkkäinen & Joo, "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age...", WACV 2021
