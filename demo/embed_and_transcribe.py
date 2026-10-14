"""Transcribe, embed and search audio files with one encoder pass per file.

Built on embed/onepass.py (OnePass): the EmbeddingGemma 2 audio tower runs once per clip; its
layers[11] output feeds V1's CTC head + beam/LM decoder (transcript) and its final output feeds
EmbeddingGemma 2's audio embedding path (768-d, L2-normalized, no prompt prefix for audio).

  transcribe FILE [FILE...] --out DIR    <name>.txt / <name>.npy / <name>.json per file
  index FOLDER --db PATH                 recursive, incremental (mtime+size) npz index
  search "query" [...] --db PATH [-k 5]  query embedded with the SearchQuery prefix; cosine top-k

Run with V1's environment:  ~/.venv-gemma-stt/bin/python demo/embed_and_transcribe.py ...
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SR = 16000
AUDIO_EXTS = {".wav", ".wave", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aif", ".aiff", ".aifc",
              ".au", ".snd", ".caf", ".w64", ".rf64"}
INDEX_FORMAT = 1
SAVE_EVERY = 25          # index: checkpoint the db every N newly embedded files


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- audio

def resample(w, sr_in, sr_out=SR):
    """Band-limited (Hann-windowed sinc, 6 zero crossings, rolloff 0.99) polyphase resampling;
    same kernel as torchaudio.functional.resample's default (torchaudio is not installed)."""
    import torch
    g = math.gcd(sr_in, sr_out)
    orig, new = sr_in // g, sr_out // g
    zeros, rolloff = 6, 0.99
    base = min(orig, new) * rolloff
    width = math.ceil(zeros * orig / base)
    idx = torch.arange(-width, width + orig, dtype=torch.float64)[None, None] / orig
    t = (torch.arange(0, -new, -1, dtype=torch.float64)[:, None, None] / new + idx) * base
    t = t.clamp(-zeros, zeros)
    window = torch.cos(t * math.pi / zeros / 2) ** 2
    t = t * math.pi
    kern = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t) * window * (base / orig)
    x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))[None, None]
    x = torch.nn.functional.pad(x, (width, width + orig))
    y = torch.nn.functional.conv1d(x, kern.to(torch.float32), stride=orig)   # [1, new, frames]
    y = y.transpose(1, 2).reshape(-1)[: math.ceil(new * len(w) / orig)]
    return y.numpy().astype(np.float32, copy=False)


def load_audio(path):
    """Any soundfile-readable file -> (float32 16 kHz mono waveform, info dict)."""
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    channels = w.shape[1]
    w = w.mean(axis=1) if channels > 1 else w[:, 0]
    info = {"sample_rate_in": int(sr), "channels_in": int(channels),
            "resampled": sr != SR, "downmixed": channels > 1}
    if sr != SR:
        w = resample(w, sr)
    return np.ascontiguousarray(w, dtype=np.float32), info


def find_audio(folder):
    out = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs.sort()
        for f in sorted(files):
            if Path(f).suffix.lower() in AUDIO_EXTS and not f.startswith("."):
                out.append(os.path.abspath(os.path.join(root, f)))
    return out


# ----------------------------------------------------------------------------- model

class Engine:
    """Lazy OnePass wrapper: the model loads only when a file or query actually needs it."""

    def __init__(self, device, dtype):
        self.device, self.dtype, self._op = device, dtype, None

    @property
    def op(self):
        if self._op is None:
            import torch
            import transformers
            from embed.onepass import OnePass
            transformers.utils.logging.disable_progress_bar()
            log(f"loading OnePass (EmbeddingGemma 2 + V1 head/decoder) on {self.device}, {self.dtype} ...")
            t0 = time.perf_counter()
            self._op = OnePass(device=self.device, dtype=getattr(torch, self.dtype))
            log(f"loaded in {time.perf_counter() - t0:.1f} s")
        return self._op

    def meta(self):
        """OnePass's embedding metadata plus the V1 transcript components, for the outputs/index."""
        import common                                   # coreml/common.py, on sys.path via onepass
        m = dict(self.op.meta)
        m["transcriber"] = {"name": "V1", "tap": "audio_tower.layers[11]", "head": common.HEAD.name,
                            "head_sha256": common.V1_SHA256[common.HEAD], "lm": common.LM.name,
                            "lm_sha256": common.V1_SHA256[common.LM], "decoder": dict(common.V1_DECODER)}
        return m

    def process(self, path):
        """One encoder pass: transcript + embedding for one file."""
        t0 = time.perf_counter()
        wav, info = load_audio(path)
        t_load = time.perf_counter() - t0
        dur = len(wav) / SR
        cap = self.op.meta.get("max_embed_seconds")
        if cap and dur > cap:
            log(f"note: {path}: {dur:.0f} s; the embedding covers the first {cap:.0f} s "
                f"(official feature-extractor limit), the transcript covers all of it")
        t1 = time.perf_counter()
        r = self.op.transcribe_and_embed(wav)
        t_model = time.perf_counter() - t1
        emb = np.asarray(r["embedding"], dtype=np.float32).reshape(-1)
        timings = {"load_audio_s": t_load, "onepass_s": t_model,
                   **{k: float(v) for k, v in r.get("timings", {}).items() if k != "audio_s"}}
        return {"text": r["text"], "embedding": emb, "duration_s": dur, "timings": timings, **info}


# ----------------------------------------------------------------------------- formatting

def term_width(default=120):
    return shutil.get_terminal_size((default, 24)).columns if sys.stdout.isatty() else default


def clip(s, n):
    return s if len(s) <= n else s[: max(n - 3, 0)].rstrip() + "..."


def snippet(text, query, n):
    """Transcript excerpt of at most n chars, centred on the query if it occurs verbatim."""
    q = query.lower().strip()
    i = text.lower().find(q) if q else -1
    if len(text) <= n or i < 0 or i + len(q) <= n - 3:
        return clip(text, n)
    start = max(0, i - max(n - 6 - len(q), 0) // 2)
    start = text.rfind(" ", 0, start + 1) + 1                       # back up to a word start
    return clip(("..." if start else "") + text[start:], n)


def table(headers, rows, aligns, width=None):
    """Aligned plain-text table; the last column is fitted to `width`. A last-column cell may be
    a callable n -> str (e.g. a snippet that wants to know how much room it gets)."""
    def last(c, n):
        return c(n) if callable(c) else clip(str(c), n)
    cells = [[str(c) for c in r[:-1]] + [r[-1]] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers[:-1])]
    full = max([len(headers[-1])] + [len(last(r[-1], 10 ** 9)) for r in cells])
    room = width - sum(w) - 2 * len(w) if width else full
    w.append(max(len(headers[-1]), min(full, room)))
    cells = [r[:-1] + [last(r[-1], w[-1])] for r in cells]

    def line(r):
        return "  ".join(c.rjust(w[i]) if aligns[i] == ">" else c.ljust(w[i])
                         for i, c in enumerate(r)).rstrip()
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in cells])


def short_path(p):
    home = str(Path.home())
    return "~" + p[len(home):] if p.startswith(home + os.sep) else p


# ----------------------------------------------------------------------------- transcribe

def cmd_transcribe(a):
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    eng, used, rows, failed = Engine(a.device, a.dtype), set(), [], 0
    for f in a.files:
        path = os.path.abspath(os.path.expanduser(f))
        name, k = Path(path).stem, 2
        while name in used:
            name, k = f"{Path(path).stem}-{k}", k + 1
        used.add(name)
        try:
            r = eng.process(path)
        except Exception as e:                            # keep going on unreadable files
            log(f"error: {f}: {type(e).__name__}: {e}")
            failed += 1
            continue
        (out / f"{name}.txt").write_text(r["text"] + "\n")
        np.save(out / f"{name}.npy", r["embedding"])
        total = r["timings"]["load_audio_s"] + r["timings"]["onepass_s"]
        info = {"file": path, "transcript": r["text"], "words": len(r["text"].split()),
                "embedding_file": f"{name}.npy", "embedding_dim": int(r["embedding"].size),
                "embedding_norm": float(np.linalg.norm(r["embedding"])),
                "duration_s": round(r["duration_s"], 3),
                **{k: r[k] for k in ("sample_rate_in", "channels_in", "resampled", "downmixed")},
                "timings_s": {k: round(v, 4) for k, v in r["timings"].items()},
                "real_time_factor": round(total / r["duration_s"], 4) if r["duration_s"] else None,
                "model": eng.meta()}
        (out / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
        rows.append([name, f"{r['duration_s']:.1f}", f"{total:.2f}", r["text"]])
    if rows:
        print(table(["file", "dur s", "time s", "transcript"], rows, "<>><", term_width()))
        print(f"\n{len(rows)} file(s) -> {short_path(str(out))}/<name>.{{txt,npy,json}}"
              + (f"; {failed} failed" if failed else ""))
    return 1 if failed else 0


# ----------------------------------------------------------------------------- index

def load_db(db):
    if not db.exists():
        return None
    z = np.load(db, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("index_format") != INDEX_FORMAT:
        raise SystemExit(f"{db}: unsupported index format {meta.get('index_format')}")
    entries = {}
    for i, p in enumerate(z["paths"]):
        entries[str(p)] = {"vector": z["vectors"][i], "text": str(z["transcripts"][i]),
                           "duration_s": float(z["durations"][i]), "mtime_ns": int(z["mtimes_ns"][i]),
                           "size": int(z["sizes"][i])}
    return meta, entries


def save_db(db, meta, entries):
    paths = sorted(entries)
    dim = int(meta.get("embedding_dim", 768))
    vec = (np.stack([entries[p]["vector"] for p in paths]).astype(np.float32)
           if paths else np.zeros((0, dim), np.float32))
    meta = dict(meta, n_files=len(paths), updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_name(db.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, paths=np.array(paths, dtype=str), vectors=vec,
                 transcripts=np.array([entries[p]["text"] for p in paths], dtype=str),
                 durations=np.array([entries[p]["duration_s"] for p in paths], np.float64),
                 mtimes_ns=np.array([entries[p]["mtime_ns"] for p in paths], np.int64),
                 sizes=np.array([entries[p]["size"] for p in paths], np.int64),
                 meta=np.array(json.dumps(meta)))
    os.replace(tmp, db)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def cmd_index(a):
    folder = os.path.abspath(os.path.expanduser(a.folder))
    if not os.path.isdir(folder):
        raise SystemExit(f"not a directory: {a.folder}")
    files = find_audio(folder)
    if not files:
        raise SystemExit(f"no audio files ({', '.join(sorted(AUDIO_EXTS))}) under {a.folder}")
    db = Path(a.db).expanduser()
    eng = Engine(a.device, a.dtype)
    old = load_db(db)
    meta, entries = old if old else ({}, {})
    from embed import onepass                     # constants only; the model loads lazily
    want = {"model_id": onepass.MODEL_ID, "revision": onepass.MODEL_REVISION, "dtype": a.dtype}
    stale = [k for k, v in want.items() if meta and meta.get(k) != v]
    if stale:
        log(f"{short_path(str(db))} was built with different {', '.join(stale)}: rebuilding it from scratch")
        meta, entries = {}, {}
    under = folder.rstrip(os.sep) + os.sep
    stats = {f: os.stat(f) for f in files}
    removed = [p for p in entries if p.startswith(under) and p not in stats]
    for p in removed:
        del entries[p]
    todo = [f for f in files if a.force or f not in entries
            or entries[f]["mtime_ns"] != stats[f].st_mtime_ns or entries[f]["size"] != stats[f].st_size]
    unchanged = len(files) - len(todo)
    log(f"{short_path(folder)}: {len(files)} audio file(s): {len(todo)} to embed, {unchanged} unchanged, "
        f"{len(removed)} removed from index")
    if todo:
        meta = {"index_format": INDEX_FORMAT, "created": meta.get("created", now()), **eng.meta(),
                "roots": sorted(set(meta.get("roots", [])) | {folder})}
    t_start, audio_s, failed, done = time.perf_counter(), 0.0, 0, 0
    pw = max((len(os.path.relpath(f, folder)) for f in todo), default=0)
    for i, f in enumerate(todo, 1):
        try:
            r = eng.process(f)
        except Exception as e:                            # keep going on unreadable files
            log(f"error: {f}: {type(e).__name__}: {e}")
            failed += 1
            continue
        st = stats[f]
        entries[f] = {"vector": r["embedding"], "text": r["text"], "duration_s": r["duration_s"],
                      "mtime_ns": st.st_mtime_ns, "size": st.st_size}
        audio_s += r["duration_s"]
        done += 1
        log(f"[{i:>{len(str(len(todo)))}}/{len(todo)}] {os.path.relpath(f, folder):<{pw}}  "
            f"{r['duration_s']:5.1f} s audio  {r['timings']['onepass_s']:5.2f} s  {clip(r['text'], 50)}")
        if done % SAVE_EVERY == 0:
            save_db(db, meta, entries)
    if done or removed:
        save_db(db, meta, entries)
    wall = time.perf_counter() - t_start
    total_s = sum(e["duration_s"] for e in entries.values())
    print(f"index   {short_path(str(db))}")
    print(f"folder  {short_path(folder)}")
    print(f"files   {len(entries)} indexed ({total_s / 60:.1f} min of audio); this run: {done} embedded, "
          f"{unchanged} unchanged, {len(removed)} removed" + (f", {failed} failed" if failed else ""))
    if done:
        print(f"time    {wall:.1f} s for {audio_s:.1f} s of audio (model load excluded)")
    return 1 if failed else 0


# ----------------------------------------------------------------------------- search

def cmd_search(a):
    db = Path(a.db).expanduser()
    old = load_db(db)
    if not old or not old[1]:
        raise SystemExit(f"{db}: no index (run `index FOLDER --db {a.db}` first)")
    meta, entries = old
    paths = sorted(entries)
    vec = np.stack([entries[p]["vector"] for p in paths]).astype(np.float32)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    eng = Engine(a.device or meta.get("device", "mps"), meta.get("dtype", "bfloat16"))
    q = eng.op.embed_queries(list(a.query))
    q = np.asarray(q, dtype=np.float32).reshape(len(a.query), -1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    roots = [r.rstrip(os.sep) + os.sep for r in meta.get("roots", [])]
    base = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])

    def rel(p):
        for r in roots:
            if p.startswith(r):
                return p[len(r):]
        return os.path.relpath(p, base)

    k = min(a.k, len(paths))
    width = term_width()
    print(f"index: {short_path(str(db))}  ({len(paths)} files, {vec.shape[1]}-d, "
          f"query prefix '{meta.get('query_prefix', '')}')")
    for qi, text in enumerate(a.query):
        s = vec @ q[qi]
        top = np.argsort(-s, kind="stable")[:k]
        rows = [[str(r + 1), f"{s[j]:.3f}", f"{entries[paths[j]]['duration_s']:.1f}", rel(paths[j]),
                 (lambda n, t=entries[paths[j]]["text"]: snippet(t, text, n))] for r, j in enumerate(top)]
        print(f'\nquery: "{text}"')
        print(table(["#", "score", "dur s", "file", "transcript"], rows, ">>><<", width))
    return 0


# ----------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.split("\n\n")[2:3]))
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="mps", help="torch device for OnePass (default mps)")
    common.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                        help="model dtype (default bfloat16; float16 is unsafe for this model)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", parents=[common], help="transcript + embedding per file")
    t.add_argument("files", nargs="+", help="audio files (any format soundfile reads)")
    t.add_argument("--out", required=True, help="output directory")
    t.set_defaults(fn=cmd_transcribe)

    i = sub.add_parser("index", parents=[common], help="embed every audio file under a folder")
    i.add_argument("folder")
    i.add_argument("--db", required=True, help="index file (.npz)")
    i.add_argument("--force", action="store_true", help="re-embed every file")
    i.set_defaults(fn=cmd_index)

    s = sub.add_parser("search", help="top-k files for one or more text queries")
    s.add_argument("query", nargs="+", help="text query (several queries share one model load)")
    s.add_argument("--db", required=True, help="index file written by `index`")
    s.add_argument("-k", type=int, default=5, help="results per query (default 5)")
    s.add_argument("--device", default=None, help="torch device (default: the index's device)")
    s.set_defaults(fn=cmd_search)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
