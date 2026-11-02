# Dataset Access Status

The full verified decision, rationale and licence details are in `dataset_decision.md`.
This file only tracks access status. Update the dates as you go.

| Dataset | Process | Status | Date |
|---|---|---|---|
| UTA-RLDD | Direct Google Drive download (10 zips) + consent spreadsheet | [ ] downloaded | |
| NTHU-DDD | Licence PDF → HoD/lab-director signature → email `cvlablai636@my.nthu.edu.tw`, subject "Dataset on Driver Drowsiness Detection from Video" | [ ] form signed · [ ] emailed · [ ] link received · [ ] downloaded | |
| YawDD | Free IEEE account → IEEE DataPort download (4.94 GB) + Participants Information | [ ] downloaded | |
| FFHQ | `download_ffhq.py --json --thumbs` first; selected 1024px images later | [ ] metadata + thumbs · [ ] selected subset | |
| FairFace | Images (pad 0.25) + labels + pretrained classifier from GitHub | [ ] downloaded | |

## NTHU-DDD email (send with the signed PDF attached)

> **To:** cvlablai636@my.nthu.edu.tw
> **Subject:** Dataset on Driver Drowsiness Detection from Video
>
> Dear NTHU Computer Vision Lab,
>
> I am a Computer Science undergraduate at BITS Pilani, Dubai Campus, working on a
> non-commercial research project on fairness in driver drowsiness detection under the
> supervision of [supervisor name]. Please find attached the NTHU-DDD Dataset License
> Agreement, signed by [name, title].
>
> I would be grateful for access to the dataset. We will cite Weng, Lai & Lai (ACCVW 2016)
> in all resulting work and will comply fully with the terms of use.
>
> Best regards,
> Aaryan
> BITS Pilani, Dubai Campus

## If NTHU-DDD access stalls
Don't switch to the Kaggle mirrors (no subject IDs or videos, and they breach the licence).
UTA-RLDD + YawDD are enough to start Phases 1–2. NTHU-DDD is mainly needed for the cue templates
(YawDD partly covers yawns) and the night-IR stress test (Phase 5, weeks away).
