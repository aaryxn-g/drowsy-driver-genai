# Synthetic Drowsy-Driver Data Generation & Fatigue Detection

Undergraduate research project — Aaryan, BITS Pilani Dubai Campus.

**Start here:**
1. `PROJECT_TRACKER.md` — live status, decision log, blockers. Update every session.
2. `docs/full_project_plan.md` — the full phase-by-phase plan (why each step exists).
3. `docs/CLAUDE_CODE_PROMPTS.md` — ready-to-paste prompts for running each phase with
   Claude Code on your own machine (GPU + dataset access required from Phase 1 onward).
4. `CLAUDE.md` — project context auto-loaded by Claude Code sessions in this repo.

**This week's action items (Phase 0):**
- [ ] Send NTHU-DDD and UTA-RLDD access requests — see `phase0_scoping/dataset_access_requests.md`
- [ ] Decide compute plan — see `phase0_scoping/compute_plan.md`
- [ ] Keep extending `phase0_scoping/annotated_bibliography.md`
- [ ] Check IRB/ethics requirements with your supervisor/department

## Repo layout
```
phase0_scoping/        literature, dataset access requests, compute plan
phase1_baseline/       baseline detector: data, models, results
phase2_generator/      GAN/diffusion reenactment pipeline + pilot
phase3_synthetic_dataset/  generated dataset (raw + balanced) + QC
phase4_retraining_sweep/   real+synthetic ratio sweep (centerpiece results)
phase5_stress_test/    night/IR, occlusion, cross-dataset evals
phase6_responsible_ai/ fairness, privacy, false-alarm, datasheet
phase7_writeup/        ablations, final report
docs/                  plan + prompts library
```
