cd /home/ubuntu/16831/16831_IRL_Project
python - <<'PY'
import glob, os, json, datetime
rows = []
for d in glob.glob("wandb/run-*"):
    s = os.path.join(d, "files", "wandb-summary.json")
    if not os.path.exists(s):
        continue
    try:
        data = json.load(open(s))
    except Exception:
        continue
    diag = [k for k in data if k.startswith("diag/")]
    if not diag:
        continue
    mtime = os.path.getmtime(d)
    rows.append((mtime, d, data.get("step", data.get("_step", None)), len(diag)))

rows.sort(reverse=True)
for mtime, d, step, n in rows[:20]:
    ts = datetime.datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {d} | step={step} | diag_keys={n}")
PY
