"""Stage 2 of the Phase 2 pilot: four fictional source identities from SDXL.

These are generated people who do not exist. No real photograph is used as a source
identity anywhere in this project.

Every candidate is written with its prompt, negative prompt, seed and sampler settings to
`identities.json`, so any image can be regenerated exactly on the same software stack.
The chosen candidate per identity is recorded separately, with the reason, by
`choose_identities` after inspection.

Run in phase2_generator/.venv (diffusers), on one GPU:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gen.make_identities
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "pilot" / "identities"
MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

COMMON = ("photorealistic studio portrait photograph, frontal head-and-shoulders, looking "
          "straight into the camera, neutral relaxed expression, mouth closed, eyes fully "
          "open, no glasses, soft even diffuse lighting, plain light grey background, "
          "sharp focus, 85mm lens, natural skin texture")
NEGATIVE = ("closed eyes, squinting, half-closed eyes, sunglasses, glasses, smiling, teeth, "
            "open mouth, hat, hair over eyes, tilted head, profile, three-quarter view, "
            "harsh shadows, rim light, dramatic lighting, cartoon, illustration, painting, "
            "3d render, anime, blurry, watermark, text, jewelry")

# Target ITA band from the 2026-11-28 collapse rules; checked by measurement afterwards,
# not assumed from the prompt.
IDENTITIES = {
    "syn01": {"target_band": "Light", "gender_presentation": "female_presenting",
              "prompt": "a 30-year-old woman with very fair light skin, " + COMMON},
    "syn02": {"target_band": "Medium", "gender_presentation": "male_presenting",
              "prompt": "a 35-year-old man with medium olive-tan skin, " + COMMON},
    "syn03": {"target_band": "Dark", "gender_presentation": "female_presenting",
              "prompt": "a 28-year-old woman with deep dark brown skin, " + COMMON},
    "syn04": {"target_band": "Dark", "gender_presentation": "male_presenting",
              "prompt": "a 40-year-old man with dark brown skin, " + COMMON},
}
SEEDS = (0, 1, 2, 3)
SETTINGS = {"num_inference_steps": 30, "guidance_scale": 6.5, "height": 1024, "width": 1024}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    import diffusers
    import torch
    from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; refusing to run SDXL on CPU")
    # V100 (sm_70): fp16 only, bf16 is not supported.
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    a.out.mkdir(parents=True, exist_ok=True)

    records = []
    for ident, spec in IDENTITIES.items():
        for seed in SEEDS:
            g = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(prompt=spec["prompt"], negative_prompt=NEGATIVE, generator=g,
                       **SETTINGS).images[0]
            name = f"{ident}_{seed}.png"
            img.save(a.out / name)
            records.append({"identity_id": ident, "seed": seed, "file": name, **spec,
                            "negative_prompt": NEGATIVE})
            print(f"  {name}")

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "variant": "fp16", "dtype": "float16",
        "scheduler": type(pipe.scheduler).__name__,
        "generator_device": "cuda", "settings": SETTINGS,
        "versions": {"diffusers": diffusers.__version__, "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0)},
        "note": "Generated people who do not exist. Seeds are per-candidate; CUDA RNG "
                "makes exact regeneration depend on GPU model and library versions.",
        "candidates": records,
    }
    (a.out / "identities.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
