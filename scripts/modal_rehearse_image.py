# Rehearse a published agent image on Modal's amd64 CPUs - the fast
# alternative to a GitHub Actions cycle when only the runtime needs
# validating (the judge VM is linux/amd64; macOS Docker runs amd64 under
# QEMU, which is untrustworthy for llama.cpp decode).
#
#   source ~/config/.env
#   uvx modal run scripts/modal_rehearse_image.py --tag sha-e591624-remote
#
# Caveat: Modal's cpu= is a request, not a hard cgroup cap like the judge's
# --cpus 2, so wall clock here is optimistic. Correctness, token counts and
# answer structure are representative; use CI for hard-capped timing.
import json
import os
import time

import modal

app = modal.App("yassai-image-rehearsal")


@app.local_entrypoint()
def main(tag: str = "sha-e591624-remote", tasks: str = "testdata/downloads_tasks_golden.json",
         label: str = "", no_remote: bool = False):
    ref = f"ghcr.io/ashaibani/yassai:{tag}"
    with open(tasks) as f:
        raw = json.load(f)
    task_list = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in raw]
    env = {
        "AGENT_MODEL": "accounts/fireworks/models/minimax-m3",
        "CALLBACK_LABEL": label or f"modal-rehearsal @ {tag}",
    }
    if no_remote:
        env["AGENT_NO_REMOTE"] = "1"
    else:
        key = os.environ.get("FIREWORKS_API_KEY", "")
        if not key:
            raise SystemExit("FIREWORKS_API_KEY not set (source ~/config/.env)")
        env["FIREWORKS_API_KEY"] = key

    # Clear the ENTRYPOINT (/yassai) and hold the sandbox open with a sleep:
    # with no command the entrypoint runs immediately, exits on the missing
    # /input/tasks.json, and the container is gone before anything is written.
    image = modal.Image.from_registry(ref).entrypoint([])
    sb = modal.Sandbox.create(
        "sleep", "1200",
        image=image,
        app=app,
        cpu=2.0,
        memory=4096,
        timeout=1200,
    )
    try:
        import base64
        payload = base64.b64encode(json.dumps(task_list).encode()).decode()
        envstr = " ".join(f"{k}='{v}'" for k, v in env.items())
        # The sandbox file API has proven flaky (FileNotFoundError on open);
        # everything rides the shell instead: tasks in via base64, results
        # out via stdout sentinels.
        script = (
            "mkdir -p /input /output && "
            f"echo {payload} | base64 -d > /input/tasks.json && "
            f"{envstr} /yassai 2>/tmp/stderr.txt; "
            "echo; echo ===STDERR-TAIL===; tail -25 /tmp/stderr.txt 2>/dev/null; "
            "echo ===RESULTS===; cat /output/results.json 2>/dev/null; echo; "
            "echo ===METRICS===; cat /output/metrics.json 2>/dev/null; echo"
        )
        start = time.time()
        p = sb.exec("/bin/sh", "-c", script)
        out_text = "".join(p.stdout)
        p.wait()
        wall = time.time() - start
        print(out_text[:out_text.find("===RESULTS===")])
        print(f"=== wall: {wall:.0f}s (Modal cpu=2 request, optimistic vs judge) ===")
        try:
            results_raw = out_text.split("===RESULTS===", 1)[1].split("===METRICS===", 1)
            results = json.loads(results_raw[0].strip())
            metrics = json.loads(results_raw[1].strip())
        except Exception as e:
            print("could not parse outputs:", e)
            print(out_text[-2000:])
            return
        empty = [r["task_id"] for r in results if not r["answer"].strip() or r["answer"] == "Unable to answer."]
        print(f"answers={len(results)} local={metrics.get('local_answers')} calls={metrics.get('calls')} "
              f"tokens={metrics.get('total_tokens')} fallbacks={metrics.get('fallbacks')} empty={empty}")
        out = f"/tmp/modal-rehearsal-{tag.replace('/', '-')}-results.json"
        with open(out, "w") as f:
            json.dump(results, f)
        print("results written:", out)
    finally:
        sb.terminate()
