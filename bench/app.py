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


def build_app(model, beam):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, Response

    app = FastAPI()
    busy = asyncio.Lock()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/replay.pcm")
    def replay_pcm(path: str):
        """A 16 kHz mono audio file as little-endian int16, the format the page's mic capture sends."""
        from common import load_audio
        pcm = (np.clip(load_audio(path), -1, 1) * 32767).astype("<i2")
        return Response(pcm.tobytes(), media_type="application/octet-stream")

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
    print(f"[bench] step model + LM loaded in {time.time() - t0:.1f} s; open http://localhost:{args.port}", flush=True)
    uvicorn.run(build_app(model, beam), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
