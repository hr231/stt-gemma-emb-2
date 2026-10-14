#!/usr/bin/env python3
"""
Terminal dashboard for head.py / probe.py training logs (the files written with `| tee`).

  python3 watch.py                                   # follows the newest log in ~/gemma_stt/logs
  python3 watch.py ~/gemma_stt/bench                 # follows the newest log in that directory
  python3 watch.py ~/gemma_stt/logs/head_R4_fast.log --interval 5
  python3 watch.py LOG --once                        # print one frame and exit

Shows progress, an estimate for the running epoch, ETA, best/last dev WER, loss and WER
sparklines, the last epochs, power source (macOS) and whether a training process is alive.
Stdlib only; Ctrl-C to quit.
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HEADER = re.compile(r"^=== (\S+)\s+\((.*)\)\s*$")
EPOCH = re.compile(r"^\s+ep\s+(\d+) \| loss\s+([\d.]+) \|(.*)$")
PROBLEM = re.compile(r"Traceback|Error|WARNING|\bnan\b|non-finite")
SPARK = "▁▂▃▄▅▆▇█"
BAR_W = 32
W = 78
RED, GREEN, OFF = "\033[31m", "\033[32m", "\033[0m"


def parse(lines):
    runs, problems = [], []
    for line in lines:
        if line.startswith("=== summary"):
            for r in runs:
                r["done"] = True
            continue
        m = HEADER.match(line)
        if m:
            info = m.group(2)
            spe = re.search(r"(\d+) steps/epoch", info)
            tot = re.search(r"(\d+) steps(?!/)", info)
            runs.append({"name": m.group(1), "info": info, "epochs": [], "done": False,
                         "total_epochs": int(tot.group(1)) // int(spe.group(1)) if spe and tot else None})
            continue
        m = EPOCH.match(line)
        if m and runs:
            rest = m.group(3)

            def num(pat):
                x = re.search(pat, rest)
                return float(x.group(1)) if x else None

            train = num(r"train\s+([\d.]+)s")
            runs[-1]["epochs"].append({
                "ep": int(m.group(1)), "loss": float(m.group(2)),
                "wer": num(r"dev WER\s+([\d.]+)%"), "blank": num(r"blank\s+([\d.]+)%"),
                "train": train if train is not None else num(r"\|\s+([\d.]+)s"),   # probe.py: one time
                "eval": num(r"eval\s+([\d.]+)s"), "best": rest.rstrip().endswith("*")})
            continue
        if PROBLEM.search(line) and "absl::InitializeLog" not in line:
            problems.append(line.rstrip())
    return runs, problems


def spark(vals, width=48):
    vals = [v for v in vals if v is not None][-width:]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def bar(frac):
    n = int(round(max(0.0, min(1.0, frac)) * BAR_W))
    return "█" * n + "░" * (BAR_W - n)


def power():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    src = "AC" if "AC Power" in out else "BATTERY" if "Battery Power" in out else "?"
    m = re.search(r"(\d+)%;\s*([^;]+);", out)
    return f"{src} {m.group(1)}% {m.group(2).strip()}" if m else src


def training_procs():
    try:
        out = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [l.strip() for l in out.splitlines()
            if re.search(r"python\S*\s.*\b(head|probe|cache_feats)\.py", l) and "caffeinate" not in l]


def fmt_dur(s):
    return f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f}s"


def render(path, args):
    text = path.read_text(errors="replace") if path.exists() else ""
    runs, problems = parse(text.splitlines())
    now = time.time()
    since = now - (path.stat().st_mtime if path.exists() else now)   # time since the last log line
    pw, procs = power(), training_procs()
    out = [f" training monitor · {path}".ljust(W - 9) + time.strftime("%H:%M:%S"),
           f" power: {pw or 'n/a'}   process: "
           + (f"running (pid {procs[0].split()[0]})" if procs else "no training process")]
    if pw and pw.startswith("BATTERY"):
        out.append(f" {RED}! on battery: plug in, or the Mac may sleep or die mid-run{OFF}")
    out.append("─" * W)
    if not runs:
        out.append(" waiting for a run header (=== ...) in the log")
        return "\n".join(out)

    for r in runs[:-1]:
        evals = [e for e in r["epochs"] if e["wer"] is not None]
        if evals:
            b = min(evals, key=lambda e: e["wer"])
            out.append(f" {r['name']:<6} {'done' if r['done'] else 'ended'}: best dev WER {b['wer']:.2f}% "
                       f"@ epoch {b['ep']} ({len(r['epochs'])} epochs)")
    r = runs[-1]
    eps = r["epochs"]
    total = args.epochs or r["total_epochs"]
    out.append(f" run    {r['name']}  ({r['info']})"[:W])
    if total:
        out.append(f" epochs [{bar(len(eps) / total)}] {len(eps)}/{total}")

    times = [e["train"] + (e["eval"] or 0) for e in eps if e["train"] is not None]
    ep_time = statistics.median(times[-6:]) if times else None      # median: robust to sleep outliers
    if r["done"]:
        out.append(" status finished")
    elif not procs:
        out.append(f" {RED}status: no training process and no summary line (stopped?){OFF}")
    elif ep_time:
        frac = min(since / ep_time, 0.99)
        out.append(f" ep {len(eps) + 1:<3} [{bar(frac)}] ~{100 * frac:3.0f}%  {since:4.0f}s / ~{ep_time:.0f}s")
        if since > 3 * ep_time:
            out.append(f" {RED}! no new epoch for {fmt_dur(since)}: stalled, asleep, or another "
                       f"process is the one running{OFF}")
        if total:
            left = max(0.0, (total - len(eps)) * ep_time - since)
            out.append(f" ETA    {time.strftime('%H:%M', time.localtime(now + left))} ({fmt_dur(left)} left)")
    else:
        out.append(f" ep {len(eps) + 1:<3} running for {fmt_dur(since)} (no epoch finished yet)")

    if eps:
        tr = [e["train"] for e in eps if e["train"] is not None][-6:]
        ev = [e["eval"] for e in eps if e["eval"] is not None][-6:]
        if tr:
            out.append(f" speed  train {statistics.median(tr):.1f}s/epoch (median of last {len(tr)})"
                       + (f" · eval {statistics.median(ev):.1f}s" if ev else ""))
        evals = [e for e in eps if e["wer"] is not None]
        if evals:
            b = min(evals, key=lambda e: e["wer"])
            out.append(f" best   {GREEN}dev WER {b['wer']:.2f}% @ epoch {b['ep']}{OFF}   "
                       f"last {evals[-1]['wer']:.2f}% @ {evals[-1]['ep']}   loss {eps[-1]['loss']:.3f}")
            w = [e["wer"] for e in evals]
            out.append(f" WER    {spark(w)}  {w[0]:.1f}% → {w[-1]:.1f}%")
        lo = [e["loss"] for e in eps]
        out.append(f" loss   {spark(lo)}  {lo[0]:.3f} → {lo[-1]:.3f}")
        out.append("─" * W)
        out.append("   ep     loss   dev WER   blank    train    eval")
        for e in eps[-args.rows:]:
            wer = f"{e['wer']:6.2f}%" if e["wer"] is not None else "     — "
            blank = f"{e['blank']:5.1f}%" if e["blank"] is not None else "    — "
            train = f"{e['train']:6.1f}s" if e["train"] is not None else "      "
            ev = f"{e['eval']:5.1f}s" if e["eval"] is not None else "      "
            out.append(f"  {e['ep']:3d}  {e['loss']:7.3f}   {wer}  {blank}  {train}  {ev}"
                       + (f"  {GREEN}*{OFF}" if e["best"] else ""))
    if problems:
        out.append("─" * W)
        out.append(f" {RED}{len(problems)} warning/error line(s); last: {problems[-1][:W - 34]}{OFF}")
    return "\n".join(out)


def newest_log(d):
    logs = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        sys.exit(f"no *.log in {d}")
    return logs[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", default="~/gemma_stt/logs",
                    help="log file, or a directory to follow its newest *.log (default ~/gemma_stt/logs)")
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds")
    ap.add_argument("--epochs", type=int, default=None, help="total epochs, if the run header doesn't say")
    ap.add_argument("--rows", type=int, default=10, help="epoch rows to show")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    args = ap.parse_args()

    target = Path(args.log).expanduser()
    pick = (lambda: newest_log(target)) if target.is_dir() else (lambda: target)   # dir: re-picked each refresh
    if args.once:
        print(render(pick(), args))
        return
    try:
        while True:
            frame = render(pick(), args)
            sys.stdout.write("\033[H\033[2J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
