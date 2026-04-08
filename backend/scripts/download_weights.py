#!/usr/bin/env python3
"""Download EchoMimic V2 pretrained weights to a RunPod network volume.

Run this once on your RunPod instance before starting the API:
    python scripts/download_weights.py --model-dir /workspace/models

This downloads:
    - sd-vae-ft-mse (VAE)
    - sd-image-variations-diffusers (base SD model)
    - EchoMimicV2 weights (UNets, pose encoder, audio processor, motion module)
    - Pose templates
"""

import argparse
import os
import subprocess
import sys


def run(cmd: str):
    print(f">>> {cmd}")
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/models", help="Where to save models")
    args = parser.parse_args()

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Clone EchoMimic V2 repo (for source code + configs)
    echomimic_dir = os.path.join(model_dir, "echomimic_v2")
    if not os.path.exists(echomimic_dir):
        run(f"git clone https://github.com/antgroup/echomimic_v2.git {echomimic_dir}")
    else:
        print(f"EchoMimic V2 repo already exists at {echomimic_dir}")

    # Download pretrained weights from HuggingFace
    weights_dir = os.path.join(echomimic_dir, "pretrained_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Use huggingface_hub to download
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    # SD VAE
    vae_dir = os.path.join(model_dir, "sd-vae-ft-mse")
    if not os.path.exists(vae_dir):
        print("Downloading sd-vae-ft-mse...")
        snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=vae_dir)
    else:
        print("sd-vae-ft-mse already downloaded")

    # SD Image Variations
    sd_dir = os.path.join(model_dir, "sd-image-variations-diffusers")
    if not os.path.exists(sd_dir):
        print("Downloading sd-image-variations-diffusers...")
        snapshot_download("lambdalabs/sd-image-variations-diffusers", local_dir=sd_dir)
    else:
        print("sd-image-variations-diffusers already downloaded")

    # EchoMimic V2 weights
    echo_weights = os.path.join(model_dir, "echomimic_v2_weights")
    if not os.path.exists(echo_weights):
        print("Downloading EchoMimicV2 weights...")
        snapshot_download("BadToBest/EchoMimicV2", local_dir=echo_weights)
        # Copy weights into the expected location
        for f in os.listdir(echo_weights):
            src = os.path.join(echo_weights, f)
            dst = os.path.join(weights_dir, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                print(f"  Linked {f}")
    else:
        print("EchoMimicV2 weights already downloaded")

    print("\nAll weights downloaded!")
    print(f"Model directory: {model_dir}")
    print("\nTo start the API server:")
    print(f"  ECHOMIMIC_MODEL_DIR={model_dir} uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
