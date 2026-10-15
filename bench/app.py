#!/usr/bin/env python3
"""Live test bench for V1: browser microphone -> WebSocket -> streaming step model + decoder.

    ~/.venv-coreml/bin/python bench/app.py [--port 8765]      then open http://localhost:8765

The page (bench/static/index.html) captures the microphone at 16 kHz in an AudioWorklet and sends
int16 PCM in 160 ms frames (2560 samples). Each frame feeds coreml/stream_runner.StreamingRecognizer
(stream_mel.IncrementalMel -> stream_step.StepModel on CPU_AND_NE -> decoder).

Modes:  A  V1 beam + LM (ctc-beam-decoder, subword 6-gram, V1_DECODER) at every step
        C  greedy CTC draft at every step; at each pause the segment is re-decoded with beam + LM
        LM off: greedy CTC everywhere (draft and final)
A segment ends after 0.6 s of blank output (stream_runner); its words become settled.

Word latency = time the word reached its final text on screen (audio clock + measured compute)
minus the word's end (Viterbi CTC alignment of the final ids on the segment's log-probs). Capture
and network delay in the browser are not included.
"""
import argparse
import asyncio
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "coreml"))
from common import OUT, SR, norm_text, sp  # noqa: E402
from stream_runner import BLANK, FRAME_S, BeamIds, StreamingRecognizer, load_step_model  # noqa: E402
from streaming import ctc_align_ends  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SESSIONS = OUT.parent / "bench_sessions"


class Greedy:
    """Greedy CTC ids (argmax, collapse repeats, drop blank)."""

    @staticmethod
    def ids(logp):
        if len(logp) == 0:
            return []
        a = np.asarray(logp).argmax(1)
        keep = np.ones(len(a), bool)
        keep[1:] = a[1:] != a[:-1]
        return [int(i) for i in a[keep] if i != BLANK]


def word_ends(logp, ids, start_frame):
    """End time (s, stream clock) of each word of ids, or None if the alignment or split fails."""
    te = ctc_align_ends(logp, ids, BLANK)
    if te is None:
        return None
    s = sp()
    groups = []
    for k, i in enumerate(ids):
        if s.id_to_piece(int(i)).startswith("\u2581") or not groups:
            groups.append([k])
        else:
            groups[-1].append(k)
    if len(groups) != len(norm_text(s.decode(ids)).split()):
        return None
    return [(start_frame + te[g[-1]]) * FRAME_S for g in groups]


class BenchRecognizer(StreamingRecognizer):
    """StreamingRecognizer with switchable step / final decoders (modes A and C, LM on / off)."""

    def __init__(self, model, beam):
        self.beam_lm, self.greedy = beam, Greedy()
        self.mode, self.lm = "A", True
        super().__init__(model, beam=beam)

    def configure(self, mode, lm):
        self.mode, self.lm = mode, lm
        self.beam = self._step_dec()

    def _step_dec(self):
        return self.beam_lm if (self.lm and self.mode == "A") else self.greedy

    def _close(self, cut, level, detail):
        self.beam = self.beam_lm if self.lm else self.greedy
        try:
            return super()._close(cut, level, detail)
        finally:
            self.beam = self._step_dec()


class Session:
    """One browser stream: recognizer, word timing, recorded audio and events."""

    def __init__(self, model, beam):
        self.rec = BenchRecognizer(model, beam)
        self.lock = threading.Lock()
        self.new()

    def new(self):
        self.rec.reset()
        self.audio, self.events = [], []
        self.settled = []                  # [{"w", "lat_ms", "end_s"}]
        self.shown, self.changed_at = [], []
        self.step_ms = []
        self.started = time.strftime("%Y%m%d-%H%M%S")

    def feed(self, pcm16):
        x = pcm16.astype(np.float32) / 32768.0
        with self.lock:
            self.audio.append(pcm16)
            return [self._handle(ev) for ev in self.rec.feed(x)]

    def finish(self):
        with self.lock:
            return [self._handle(ev) for ev in self.rec.finish()]

    def _handle(self, ev):
        now = ev["t_audio"] + ev["compute_ms"] / 1e3
        self.step_ms.append(ev["compute_ms"])
        if ev["kind"] == "update":
            words = ev["live"] + ([ev["pending"]] if ev["pending"] else [])
            self.changed_at = [self.changed_at[i] if i < len(self.shown) and i < len(self.changed_at)
                               and self.shown[i] == w else now for i, w in enumerate(words)]
            self.shown = words
        else:
            d = ev["detail"]
            ends = word_ends(self.rec.lp[d["start_frame"]:d["end_frame"]], d["ids"], d["start_frame"])
            for i, w in enumerate(ev["final"]):
                same = i < len(ev["replaced"]) and ev["replaced"][i] == w and i < len(self.changed_at)
                settle = self.changed_at[i] if same else now
                end = ends[i] if ends else None
                self.settled.append({"w": w, "end_s": end,
                                     "lat_ms": None if end is None else round((settle - end) * 1e3)})
            self.shown, self.changed_at = [], []
        self.events.append({k: v for k, v in ev.items() if k != "detail"} | {"detail": {
            k: v for k, v in ev["detail"].items() if k != "ids"}})
        return self.state(ev)

    def state(self, ev=None):
        lats = [w["lat_ms"] for w in self.settled if w["lat_ms"] is not None]
        steps = self.step_ms[-200:]
        return {"type": "state", "settled": self.settled[-400:],
                "tentative": self.shown, "pending_last": bool(ev and ev.get("pending")),
                "t_audio": self.rec.t_audio, "mode": self.rec.mode, "lm": self.rec.lm,
                "level": None if ev is None else ev.get("level"),
                "kind": None if ev is None else ev["kind"],
                "changed_words": None if ev is None else ev.get("changed_words"),
                "hud": {"lat_median": float(np.median(lats)) if lats else None,
                        "lat_p90": float(np.percentile(lats, 90)) if lats else None,
                        "lat_last": lats[-1] if lats else None, "n_words": len(lats),
                        "step_ms": steps[-1] if steps else None,
                        "step_p50": float(np.median(steps)) if steps else None,
                        "rtf": (float(np.median(steps)) / 160.0) if steps else None}}

    def save(self):
        with self.lock:
            d = SESSIONS / self.started
            d.mkdir(parents=True, exist_ok=True)
            pcm = np.concatenate(self.audio) if self.audio else np.zeros(0, np.int16)
            with wave.open(str(d / "audio.wav"), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SR)
                f.writeframes(pcm.tobytes())
            (d / "session.json").write_text(json.dumps({
                "started": self.started, "audio_s": len(pcm) / SR, "mode": self.rec.mode, "lm": self.rec.lm,
                "transcript": " ".join(w["w"] for w in self.settled + [{"w": t} for t in self.shown]),
                "words": self.settled, "events": self.events}, indent=1, default=float))
            return str(d)


def word_diff(ref, hyp):
    """Word alignment (Levenshtein backtrace): list of (op, ref_word, hyp_word), op in = sub del ins."""
    R, H = len(ref), len(hyp)
    d = np.zeros((R + 1, H + 1), int)
    d[:, 0], d[0, :] = np.arange(R + 1), np.arange(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], R, H
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    return ops[::-1], int(d[R, H])


def build_app(model, beam, tr):
    from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, Response
    from common import load_audio, load_list

    app = FastAPI()
    busy = asyncio.Lock()
    offline = threading.Lock()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/tools")
    def tools():
        return FileResponse(STATIC / "tools.html")

    @app.get("/replay.pcm")
    def replay_pcm(path: str):
        """A 16 kHz mono audio file as little-endian int16, the format the page's mic capture sends."""
        pcm = (np.clip(load_audio(path), -1, 1) * 32767).astype("<i2")
        return Response(pcm.tobytes(), media_type="application/octet-stream")

    def pause_chunks(wav, n_mel, min_gap_s=0.25, max_mel=3500):
        """Cut points (mel frames, on the 4-frame output grid) in the middle of every silence of at least
        min_gap_s (10 ms frame energy below the loudest frame - 35 dB); chunks longer than max_mel are
        split evenly."""
        fr = wav[: len(wav) // 160 * 160].reshape(-1, 160)
        quiet = 10 * np.log10((fr ** 2).mean(1) + 1e-10) < 10 * np.log10((fr ** 2).mean(1).max() + 1e-10) - 35
        cuts = [0]
        i, N = 0, len(quiet)
        while i < N:
            if quiet[i]:
                j = i
                while j < N and quiet[j]:
                    j += 1
                if j - i >= int(min_gap_s * 100) and i > 0 and j < N:
                    cuts.append(((i + j) // 2) // 4 * 4)
                i = j
            else:
                i += 1
        cuts.append(n_mel)
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            k = -(-(b - a) // max_mel)
            edges = [a + (b - a) * t // k // 4 * 4 for t in range(k)] + [b]
            out += [(x, y) for x, y in zip(edges[:-1], edges[1:]) if y > x]
        return out

    def run_offline(path, min_gap_s=1.0):
        """Offline Core ML release build: read file -> mel -> cut at pauses -> each chunk in the smallest
        window that fits -> beam + LM per chunk (as the T10 benchmark, which runs utterance by utterance)."""
        from common import mel
        with offline:
            t0 = time.perf_counter()
            wav = load_audio(path)
            m, n = mel(wav)
            m = m[:n]
            t1 = time.perf_counter()
            chunks = pause_chunks(wav, n, min_gap_s)
            lps = [tr.cml.predict(m[a:b], b - a)[0] for a, b in chunks]
            t2 = time.perf_counter()
            text = " ".join(t for t in (norm_text(tr.decoder.decode(lp)) for lp in lps) if t)
            t3 = time.perf_counter()
        audio_s = len(wav) / SR
        return {"text": text, "audio_s": audio_s, "server_s": t3 - t0, "mel_s": t1 - t0, "model_s": t2 - t1,
                "decode_s": t3 - t2, "rtfx": audio_s / (t3 - t0), "segments": len(chunks), "min_gap_s": min_gap_s,
                "path": path}

    @app.post("/api/transcribe")
    async def api_transcribe(body: dict = Body(...)):
        return await asyncio.to_thread(run_offline, body["path"], float(body.get("min_gap_s", 1.0)))

    @app.get("/api/spot")
    async def api_spot(split: str = "test-clean", seed: int = 0, draw: int = 0):
        """Draw number `draw` (0-based) of random.Random(seed) over the split's utterances; no filtering."""
        import random
        utts = [u for u in load_list("release" if split.startswith("test") else "full") if u["split"] == split]
        rng = random.Random(seed)
        for _ in range(draw + 1):
            u = rng.choice(utts)
        r = await asyncio.to_thread(run_offline, u["audio"])
        ref = norm_text(u.get("text") or u.get("ref")).split()
        ops, edits = word_diff(ref, r["text"].split())
        return r | {"id": u["id"], "split": split, "seed": seed, "draw": draw, "ref": " ".join(ref), "ops": ops,
                    "edits": edits, "ref_words": len(ref), "wer": edits / max(1, len(ref)), "n_utts": len(utts)}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        if busy.locked():
            await sock.send_json({"type": "error", "message": "another tab is streaming; close it first"})
            await sock.close()
            return
        async with busy:
            s = Session(model, beam)
            await sock.send_json({"type": "ready"})
            try:
                while True:
                    msg = await sock.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        pcm = np.frombuffer(msg["bytes"], dtype="<i2")
                        for st in await asyncio.to_thread(s.feed, pcm):
                            await sock.send_json(st)
                        continue
                    cmd = json.loads(msg["text"])
                    if cmd["cmd"] == "config":
                        s.rec.configure(cmd.get("mode", "A"), bool(cmd.get("lm", True)))
                        await sock.send_json(s.state())
                    elif cmd["cmd"] == "stop":
                        for st in await asyncio.to_thread(s.finish):
                            await sock.send_json(st)
                        await sock.send_json(s.state() | {"type": "stopped"})
                    elif cmd["cmd"] == "reset":
                        mode, lm = s.rec.mode, s.rec.lm
                        s.new()
                        s.rec.configure(mode, lm)
                        await sock.send_json(s.state())
                    elif cmd["cmd"] == "save":
                        path = await asyncio.to_thread(s.save)
                        await sock.send_json({"type": "saved", "path": path})
            except WebSocketDisconnect:
                pass

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--compute-units", default="CPU_AND_NE")
    args = ap.parse_args()
    import uvicorn
    t0 = time.time()
    model = load_step_model("step", compute_units=args.compute_units)
    beam = BeamIds()
    warm = BenchRecognizer(model, beam)                 # first predictions compile / warm the ANE
    for _ in warm.feed(np.zeros(SR * 2, np.float32)):
        pass
    warm.reset()
    from transcribe import Transcriber
    tr = Transcriber()                                   # offline release package (all four windows)
    for w in (1, 8, 18, 34):                            # warm every window function
        tr.transcribe(np.zeros(int(SR * w), np.float32) + 1e-4)
    print(f"[bench] step model + LM + offline release loaded in {time.time() - t0:.1f} s; open http://localhost:{args.port}", flush=True)
    uvicorn.run(build_app(model, beam, tr), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
