import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video


PROMPTS = [
    "A kinetic Rio carnival parade at night with feathered dancers, mirrored floats, spinning confetti cannons, wet reflective pavement, saturated neon magenta and cyan lights, intricate beadwork, sequins, smoke bursts, handheld camera energy, rapid costume motion, rich textures, dense crowd choreography.",
    "A bioluminescent coral reef canyon filled with schools of tropical fish, jellyfish pulses, drifting bubbles, iridescent scales, branching corals, caustic light patterns, colorful sea fans, tiny particles in the water, sweeping camera motion, extremely vivid color separation, high texture complexity.",
    "A futuristic cyberpunk street market in heavy rain, holographic signs flickering, steam vents, motorbikes splashing through puddles, LED umbrellas, animated billboards, reflective chrome stalls, people weaving through the crowd, strong motion parallax, dense fine detail, vibrant red blue green lighting.",
    "A volcanic dragon festival at dusk, giant silk dragons coiling through the sky, fireworks exploding, ash sparks, glowing lanterns, embroidered robes whipping in the wind, molten rock textures, swirling smoke, dramatic camera moves, highly dynamic scene, vivid oranges, teal shadows, rich microdetail.",
    "A glass greenhouse jungle during a thunderstorm, exotic flowers opening and closing, butterflies swarming, rain racing down the panes, lightning flashes refracting through glass, wet leaves, moss, water droplets, layered foliage motion, lush saturated greens and flowers, extremely detailed textures.",
    "A surreal desert fashion procession with mirrored fabric banners, kaleidoscopic dust devils, reflective jewelry, embroidered cloth, iridescent beetles, colorful silk ribbons, golden hour sunlight, rolling dunes, moving shadows, highly dynamic cloth simulation, dense high-frequency patterns.",
    "An ice cave with crystalline walls under aurora light, skaters carving luminous trails, drifting snow powder, glittering frost, translucent ice textures, mirrored reflections, moving spotlights, camera flying through arches and tunnels, rich blue green violet palette, intricate detail everywhere.",
    "A mechanical clockwork city square at sunrise, brass birds taking flight, gears spinning in towers, steam valves pulsing, polished copper surfaces, stained glass reflections, people in ornate coats crossing the plaza, rotating signs and moving trams, warm light, dense texture and motion.",
    "A street dance battle in a graffiti tunnel, colorful paint particles bursting on each beat, laser strips strobing, sweat droplets, patterned fabrics, chain jewelry, fast footwork, camera circling the dancers, strong contrast, saturated color splashes, fine texture and high-frequency motion.",
    "A fantasy forest ritual with glowing insects, swirling leaves, embroidered ceremonial capes, luminous mushrooms, moving fog, sparks from a fire circle, carved wooden masks, flickering torchlight, rich emerald and amber tones, layered motion, highly detailed bark, fabric, and foliage textures.",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--size", default="832*480", type=str)
    parser.add_argument("--frame-num", default=81, type=int)
    parser.add_argument("--sample-steps", default=50, type=int)
    parser.add_argument("--sample-shift", default=8.0, type=float)
    parser.add_argument("--guide-scale", default=6.0, type=float)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--videos-per-group", default=2, type=int)
    parser.add_argument("--base-seed", default=20260330, type=int)
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--t5-cpu", action="store_true", default=True)
    parser.add_argument("--device-id", default=0, type=int)
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def main():
    args = parse_args()
    setup_logging()

    os.makedirs(args.output_dir, exist_ok=True)
    config = WAN_CONFIGS["t2v-1.3B"]
    size = SIZE_CONFIGS[args.size]

    logging.info("Loading WanT2V-1.3B pipeline.")
    pipe = wan.WanT2V(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    manifest = {
        "task": "t2v-1.3B",
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "guide_scale": args.guide_scale,
        "sample_solver": args.sample_solver,
        "sample_fps": config.sample_fps,
        "videos": [],
    }

    run_started = time.time()
    for group_idx, prompt in enumerate(PROMPTS, start=1):
        group_dir = Path(args.output_dir) / f"group_{group_idx:02d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = group_dir / "prompt.txt"
        prompt_path.write_text(prompt)

        for variant_idx in range(1, args.videos_per_group + 1):
            seed = args.base_seed + group_idx * 100 + variant_idx
            stem = f"group_{group_idx:02d}_variant_{variant_idx:02d}_seed_{seed}"
            video_path = group_dir / f"{stem}.mp4"
            latent_path = group_dir / f"{stem}_latent.pt"
            meta_path = group_dir / f"{stem}.json"

            logging.info("Generating %s", stem)
            started = time.time()
            video = pipe.generate(
                prompt,
                size=size,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.guide_scale,
                seed=seed,
                offload_model=args.offload_model,
                save_latent_path=str(latent_path),
            )
            cache_video(
                tensor=video[None],
                save_file=str(video_path),
                fps=config.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            elapsed = time.time() - started
            latent_payload = torch.load(latent_path, map_location="cpu")
            record = {
                "group": group_idx,
                "variant": variant_idx,
                "seed": seed,
                "prompt": prompt,
                "video_path": str(video_path),
                "latent_path": str(latent_path),
                "video_bytes": video_path.stat().st_size,
                "latent_bytes": latent_path.stat().st_size,
                "latent_shape": list(latent_payload["latent"].shape),
                "fps": latent_payload["fps"],
                "elapsed_s": elapsed,
            }
            meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            manifest["videos"].append(record)
            logging.info(
                "Finished %s in %.1fs, video=%d bytes, latent=%d bytes",
                stem,
                elapsed,
                record["video_bytes"],
                record["latent_bytes"],
            )

    manifest["total_elapsed_s"] = time.time() - run_started
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logging.info("All generations finished. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
