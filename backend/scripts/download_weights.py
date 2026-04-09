#!/usr/bin/env python3
"""Download EchoMimic V3 Flash models to a RunPod network volume.

Run this once on your RunPod instance before starting the API:
    python scripts/download_weights.py --model-dir /workspace/models
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

    # Clone EchoMimic V3 repo (for source code: pipeline, transformer, wav2vec2 modules)
    v3_dir = os.path.join(model_dir, "echomimic_v3")
    if not os.path.exists(v3_dir):
        run(f"git clone https://github.com/antgroup/echomimic_v3.git {v3_dir}")
    else:
        print(f"EchoMimic V3 repo already exists at {v3_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        run("pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    # Wan2.1-Fun base model (VAE, text encoder, CLIP, tokenizer, config)
    wan_dir = os.path.join(model_dir, "Wan2.1-Fun-V1.1-1.3B-InP")
    if not os.path.exists(wan_dir):
        print("Downloading Wan2.1-Fun-V1.1-1.3B-InP base model...")
        snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", local_dir=wan_dir)
    else:
        print("Wan2.1-Fun base model already downloaded")

    # EchoMimic V3 transformer weights (includes flash-pro variant)
    v3_weights = os.path.join(model_dir, "EchoMimicV3")
    if not os.path.exists(v3_weights):
        print("Downloading EchoMimicV3 weights...")
        snapshot_download("BadToBest/EchoMimicV3", local_dir=v3_weights)
    else:
        print("EchoMimicV3 weights already downloaded")

    # Chinese Wav2Vec2 audio encoder (for Flash variant)
    wav2vec_dir = os.path.join(model_dir, "chinese-wav2vec2-base")
    if not os.path.exists(wav2vec_dir):
        print("Downloading chinese-wav2vec2-base audio encoder...")
        try:
            snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=wav2vec_dir)
        except Exception:
            # Fallback: try ModelScope if HuggingFace doesn't have it
            print("HuggingFace download failed, trying modelscope...")
            run(f"pip install modelscope")
            from modelscope.hub.snapshot_download import snapshot_download as ms_download
            ms_download("TencentGameMate/chinese-wav2vec2-base", cache_dir=wav2vec_dir)
    else:
        print("chinese-wav2vec2-base already downloaded")

    print("\nAll V3 Flash models downloaded!")
    print(f"Model directory: {model_dir}")
    print(f"\nTo start the API server:")
    print(f"  cd /workspace/stillalive/backend")
    print(f"  ECHOMIMIC_MODEL_DIR={model_dir} \\")
    print(f"  PYTHONPATH={v3_dir}:$PYTHONPATH \\")
    print(f"  uvicorn app.main:app --host 0.0.0.0 --port 8888")


if __name__ == "__main__":
    main()
