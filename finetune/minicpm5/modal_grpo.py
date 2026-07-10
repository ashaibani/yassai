"""Modal runner for GRPO/RLVR over the assist SFT model.

Usage:
  # Build a small verifiable prompt pool, then:
  uvx modal run finetune/minicpm5/modal_grpo.py --base-run assist-e3-r32-v6 --tag grpo1

Prototype defaults: 1 epoch, 4 generations, LoRA r=16, lr=5e-6 on H100.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "yassai-minicpm5-grpo"
REMOTE_ROOT = Path("/root/yassai")
CKPT_ROOT = Path("/checkpoints")

app = modal.App(APP_NAME)
ckpt_volume = modal.Volume.from_name("yassai-minicpm5-checkpoints", create_if_missing=False)
hf_cache = modal.Volume.from_name("yassai-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        # Qwen3.5 (model_type qwen3_5) needs transformers>=5; keep peft/trl
        # aligned with the SFT image so tool-lane GRPO can load the SFT merge.
        "transformers==5.13.0",
        "trl==0.24.0",
        "peft==0.19.1",
        "datasets==3.6.0",
        "accelerate>=1.11.0",
        "safetensors>=0.6.2",
        "sentencepiece==0.2.1",
        "protobuf==6.33.0",
        "huggingface_hub>=0.34",
    )
    .add_local_file("finetune/minicpm5/train_grpo.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/train_grpo.py"), copy=True)
    .add_local_file("finetune/minicpm5/rewards_rlvr.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/rewards_rlvr.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_assist_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_assist_behavior.py"), copy=True)
    .add_local_file("scripts/build_minicpm5_assist_data.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_teacher_raw.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"), copy=True)
    .add_local_file("finetune/minicpm5/data/assist_claude_authored.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/assist_claude_authored.jsonl"), copy=True)
    # Tool-lane clue-encoding RLVR: parametric puzzle pool with derived truths
    # (scripts/build_logic_grpo_pool.py) + the v2 builder and behaviour gate.
    .add_local_file("finetune/minicpm5/data/logic_grpo_pool.jsonl", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/data/logic_grpo_pool.jsonl"), copy=True)
    .add_local_file("scripts/build_minicpm5_sft_data_v2.py", remote_path=str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"), copy=True)
    .add_local_file("finetune/minicpm5/eval_tool_behavior.py", remote_path=str(REMOTE_ROOT / "finetune/minicpm5/eval_tool_behavior.py"), copy=True)
)


def build_grpo_prompt_pool(data_dir: Path, max_per_family: int = 40) -> Path:
    """Flatten the assist heldout+train SFT messages into GRPO {family,system,prompt} rows."""
    # Rebuild both splits so we never touch golden/live tasks.
    train_path = data_dir / "minicpm5_yassai_assist.jsonl"
    heldout_path = data_dir / "minicpm5_yassai_assist_heldout.jsonl"
    for split, out, seed in (
        ("train", train_path, 42),
        ("heldout", heldout_path, 20269998),
    ):
        cmd = [
            "python", str(REMOTE_ROOT / "scripts/build_minicpm5_assist_data.py"),
            "--teacher", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_teacher_raw.jsonl"),
            "--claude", str(REMOTE_ROOT / "finetune/minicpm5/data/assist_claude_authored.jsonl"),
            "--out", str(out), "--teacher-split", split,
        ]
        if split == "heldout":
            cmd += ["--seed", str(seed), "--ner", "24"]
        subprocess.run(cmd, check=True)

    # Prefer heldout for RL (generalisation); pad with train if thin.
    SYSTEM = {
        "code_generation": "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
        "ner": "Extract the entities in exactly the requested format, nothing else. Include every entity present; omissions are failures.",
        "sentiment": "Classify the sentiment and justify it in one accurate sentence. Never contradict the text.",
        "summarisation": "Obey the stated length and format limits exactly; cover every major theme; no preamble.",
        "factual": "Answer directly and concisely; explanations max 3 short sentences.",
        "code_fix": "State the one-line cause of the bug (what the buggy line actually does), then provide the minimal corrected function. Plain code, no fences, change only the bug.",
    }
    by_fam: dict[str, list] = {}
    for path in (heldout_path, train_path):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row.get("messages") or []
            sys = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            fam = row.get("family") or next((k for k, v in SYSTEM.items() if v == sys), "")
            if not fam or not user:
                continue
            bucket = by_fam.setdefault(fam, [])
            if len(bucket) >= max_per_family:
                continue
            if any(x["prompt"] == user for x in bucket):
                continue
            bucket.append({"family": fam, "system": SYSTEM[fam], "prompt": user})

    out = data_dir / "assist_grpo_prompts.jsonl"
    rows = [r for fam in sorted(by_fam) for r in by_fam[fam]]
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"GRPO pool: {len(rows)} prompts across { {k: len(v) for k,v in by_fam.items()} }")
    return out


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/checkpoints": ckpt_volume, "/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def grpo(base_run: str = "assist-e3-r32-v6", tag: str = "grpo1", epochs: float = 1.0,
         lr: float = 5e-6, rank: int = 16, num_generations: int = 4,
         lane: str = "assist") -> str:
    base = CKPT_ROOT / base_run / "merged_hf"
    if not base.exists():
        raise SystemExit(f"base merged model missing: {base}")
    data_dir = REMOTE_ROOT / "finetune/minicpm5/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if lane == "tool":
        # Clue-encoding RLVR: the pool ships in the image (derived truths,
        # overlap-asserted at build time by scripts/build_logic_grpo_pool.py).
        pool = data_dir / "logic_grpo_pool.jsonl"
    else:
        pool = build_grpo_prompt_pool(data_dir)
    out_dir = CKPT_ROOT / f"{lane}-grpo-{tag}"

    env = {
        "BASE_MODEL": str(base),
        "DATA": str(pool),
        "OUTPUT_DIR": str(out_dir),
        "EPOCHS": str(epochs),
        "LR": str(lr),
        "LORA_RANK": str(rank),
        "NUM_GENERATIONS": str(num_generations),
        "HF_HOME": "/cache/huggingface",
        "PYTHONPATH": str(REMOTE_ROOT / "finetune/minicpm5"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5/train_grpo.py")], check=True, env={**os.environ, **env})

    if lane == "tool":
        # Behaviour gate on a fresh held-out v2 split: GRPO must not regress
        # tool-call formats or the maths families (the toolv3 lesson).
        heldout = data_dir / "minicpm5_yassai_v2_heldout.jsonl"
        subprocess.run(
            ["python", str(REMOTE_ROOT / "scripts/build_minicpm5_sft_data_v2.py"),
             "--out", str(heldout), "--seed", "20269999"],
            check=True,
        )
        gate = "eval_tool_behavior.py"
    else:
        heldout = data_dir / "minicpm5_yassai_assist_heldout.jsonl"
        gate = "eval_assist_behavior.py"
    eval_env = {
        **env,
        "BASE_MODEL": str(base),
        "ADAPTER": str(out_dir / "adapter_final"),
        "DATA": str(heldout),
    }
    subprocess.run(["python", str(REMOTE_ROOT / "finetune/minicpm5" / gate)], check=True, env={**os.environ, **eval_env})

    ckpt_volume.commit()
    hf_cache.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(base_run: str = "assist-e3-r32-v6", tag: str = "grpo1", epochs: float = 1.0,
         lr: float = 5e-6, rank: int = 16, num_generations: int = 4, lane: str = "assist"):
    print(grpo.remote(base_run=base_run, tag=tag, epochs=epochs, lr=lr, rank=rank,
                      num_generations=num_generations, lane=lane))
