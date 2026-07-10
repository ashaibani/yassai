"""Modal runner for MiniCPM5 yassai LoRA training.

Usage:
  uvx modal run finetune/minicpm5/modal_train.py
  uvx modal run finetune/minicpm5/modal_train.py --dataset exact --epochs 14
"""

from __future__ import annotations

import subprocess
import os
from pathlib import Path

import modal


APP_NAME = "yassai-minicpm5-sft"
REMOTE_ROOT = Path("/root/yassai")
CKPT_ROOT = Path("/checkpoints")

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("yassai-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.57.3",
        "trl==0.20.0",
        "peft==0.11.1",
        "datasets==3.6.0",
        "accelerate==1.11.0",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
    )
    .add_local_file("scripts/build_minicpm5_sft_data.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data.py"), copy=True)
    .add_local_file("scripts/build_minicpm5_sft_data_v2.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"), copy=True)
    .add_local_file("scripts/build_minicpm5_assist_data.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"), copy=True)
    # Teacher cache is judge-filtered locally (needs the Fireworks key, which
    # this image must not have) and shipped in as data.
    .add_local_file("finetune/minicpm5/data/assist_teacher_raw.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"), copy=True)
    .add_local_file("finetune/minicpm5/train_trl.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/train_trl.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_tool_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_assist_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_assist_behavior.py"), copy=True)
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
)
def train(dataset: str = "v2", epochs: float = 3.0, lr: float = 1.0e-4, rank: int = 32) -> str:
    data_dir = REMOTE_ROOT / "finetune/minicpm5/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    eval_path = None
    if dataset == "assist":
        # Assistant-lane families (ner/cg parametric + judge-filtered teacher
        # rows for sentiment/summarisation/factual). Direct answers, no tool
        # contract - this adapter replaces the BASE model in the second lane.
        data_path = data_dir / "minicpm5_yassai_assist.jsonl"
        build_cmd = ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"),
                     "--teacher", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"),
                     "--out", str(data_path), "--teacher-split", "train"]
        eval_path = data_dir / "minicpm5_yassai_assist_heldout.jsonl"
        subprocess.run(
            ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"),
             "--teacher", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"),
             "--out", str(eval_path), "--seed", "20269998", "--ner", "24",
             "--teacher-split", "heldout"],
            check=True,
        )
    elif dataset == "v2":
        # Parameterised, execution-verified cases. No sample-task copies: the
        # leaderboard scores unseen variants and forbids hardcoding.
        data_path = data_dir / "minicpm5_yassai_v2.jsonl"
        build_cmd = ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"), "--out", str(data_path)]
        # Held-out split (different RNG seed -> disjoint prompts): the
        # post-train behaviour eval must measure generalisation, not recall.
        # The v2-e5 overfit shipped precisely because eval ran on train data.
        eval_path = data_dir / "minicpm5_yassai_v2_heldout.jsonl"
        subprocess.run(
            ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"),
             "--out", str(eval_path), "--seed", "20269999"],
            check=True,
        )
    else:
        data_path = data_dir / (
            "minicpm5_yassai_math_logic_exact_ablation.jsonl"
            if dataset == "exact"
            else "minicpm5_yassai_math_logic.jsonl"
        )
        build_cmd = ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data.py"), "--out", str(data_path)]
        if dataset == "exact":
            build_cmd.insert(-2, "--include-exact-downloads")
    subprocess.run(build_cmd, check=True)

    run_name = f"{dataset}-e{epochs:g}-r{rank}"
    out_dir = CKPT_ROOT / run_name
    env = {
        "BASE_MODEL": "openbmb/MiniCPM5-1B",
        "DATA": str(data_path),
        "OUTPUT_DIR": str(out_dir),
        "EPOCHS": str(epochs),
        "LR": str(lr),
        "LORA_RANK": str(rank),
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/train_trl.py")], check=True, env={**os.environ, **env})

    eval_env = {
        **env,
        "ADAPTER": str(out_dir / "adapter_final"),
    }
    if eval_path is not None:
        eval_env["DATA"] = str(eval_path)
    # The assist adapter answers directly (no tool contract), so it gets the
    # verifiable per-family behaviour gate instead of the tool-call eval.
    eval_script = "eval_assist_behavior.py" if dataset == "assist" else "eval_tool_behavior.py"
    subprocess.run(["python", str(REMOTE_ROOT / ("finetune/minicpm5/" + eval_script))], check=True, env={**os.environ, **eval_env})

    ckpt_volume.commit()
    hf_cache.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(dataset: str = "exact", epochs: float = 14.0, lr: float = 1.0e-4, rank: int = 32):
    print(train.remote(dataset=dataset, epochs=epochs, lr=lr, rank=rank))
