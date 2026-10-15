"""Streaming V1 recognizer, mode A: audio in, display events out, one step-model call per 160 ms.

    from stream_runner import StreamingRecognizer, load_step_model
    rec = StreamingRecognizer(load_step_model("step"))       # or load_step_model("standin")
    for ev in rec.feed(samples): ...         # float32 16 kHz mono, any length, repeatedly
    for ev in rec.finish(): ...              # end of stream: last chunk, flush, final decode
    rec.reset()                              # new stream (same model and decoder)

Pipeline per step j (0-based): stream_mel.IncrementalMel emits mel frames as samples arrive; once
mel frames [16j, 16j + 16) exist (sample 2560 j + 2560 has arrived) the step model gets them and
returns log-probs for output frames [4j - 4, 4j) (its 4-frame = 160 ms lookahead delay); frames < 0
are dropped. At the end of the stream (V valid mel frames, L = ceil(V / 4) output frames) finish()
runs the last partial chunk (n_valid < 16) if any, then one flush call, and keeps frames < L. The
decoder is V1's ctc-beam-decoder (common.beam_decoder: piece units, subword 6-gram, beam 32,
alpha 0.8, beta 1): mode A re-decodes the open segment's accumulated log-probs after every step that
brought new frames (about 4.6 us per frame of segment on an M-series CPU; 3.7 ms for 32 s) and shows
every word, the last one flagged as pending.

Segmentation (decoder only; the acoustic model streams on without resets): with pause_s set, the
open segment closes once its last pause_s of output frames are all blank (argmax) after at least one
non-blank frame; its beam+LM text is final ('segment_end') and joins the committed words. Leading
all-blank frames are trimmed to preroll_s. A segment longer than max_segment_s is cut in the middle
of the longest blank-argmax run in its last 5 s (at its end if there is none). pause_s=None and
max_segment_s=None give one segment per stream, whose final text is exactly V1's beam+LM decode of
the stream's log-probs.

Events (word fields are lists of lowercase words), as coreml/livecore.py:
  {'kind': 'update', 't_audio': s of audio at which this step's input was complete, 'segment',
   'committed', 'live': open segment's words but the last, 'pending': last word or None,
   'compute_ms': mel + model + decode for this step, 'window': 'step', 'level': RMS dBFS of the
   audio since the previous step, 'detail': {mel_ms, model_ms, decode_ms, calls, frames, step}}
  {'kind': 'segment_end', 'segment', 'final', 'replaced', 'changed_words', 't_audio', 'compute_ms',
   'window': 'step', 'level', 'detail': {... 'start_frame', 'end_frame', 'ids'}}

Step models: load_step_model('step') is StreamStep's coreml/stream_step.StepModel (stateful Core ML);
load_step_model('standin') is WindowStandIn, which re-runs the offline Core ML window package (default
variant 'release', the released multifunction package; cmlrun.V1CoreML) on the
accumulated mel at every step and returns the same frames (the model is causal with 4 frames of
lookahead, so those frames are final) - a correctness stand-in, much slower than a real step.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SR, WINDOWS, beam_decoder, norm_text, sp  # noqa: E402
from stream_mel import HOP, N_FEAT, IncrementalMel  # noqa: E402
from wer import edit_distance  # noqa: E402

MEL_PER_STEP = 16                 # 160 ms of new audio
OUT_PER_STEP = 4                  # output frames (40 ms) per step
DELAY = 4                         # step j returns output frames [4j - 4, 4j)
FRAME_S = 0.04
BLANK = 256
N_CLASSES = 257


def rms_db(x):
    if len(x) == 0:
        return -100.0
    return max(-100.0, float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-10)))


def step_ready_samples(j):
    """Samples needed before step j can run: mel frame 16j + 15 needs sample 160 (16j + 15) + 160."""
    return HOP * (MEL_PER_STEP * j + MEL_PER_STEP - 1) + HOP + 1


# ---------------------------------------------------------------- step models
class WindowStandIn:
    """StepModel interface on the offline Core ML windows (cmlrun.V1CoreML): every call predicts on
    the accumulated mel and returns output frames [4j - 4, 4j). Beyond 35 s it predicts on the newest
    2000 mel frames (20 s window, on the 4-frame grid): 500 output frames > the 452-frame past
    receptive field + 4 of lookahead, so the returned frames are still exact."""

    TAIL = WINDOWS["20s"]

    def __init__(self, cml):
        self.cml = cml
        self.reset()

    def reset(self):
        self.mel = np.zeros((0, N_FEAT), np.float32)
        self.calls = 0
        self.lp = np.zeros((0, N_CLASSES), np.float32)   # last prediction
        self.lp_off = 0                                    # output frame index of self.lp[0]

    def _rows(self, a):
        out = np.zeros((OUT_PER_STEP, N_CLASSES), np.float32)
        for r in range(OUT_PER_STEP):
            k = a + r - self.lp_off
            if a + r >= 0 and 0 <= k < len(self.lp):
                out[r] = self.lp[k]
        return out

    def step(self, mel16, n_valid=MEL_PER_STEP):
        self.mel = np.concatenate([self.mel, np.asarray(mel16, np.float32)[:n_valid]])
        m, off = self.mel, 0
        if len(m) > max(WINDOWS.values()):
            off = (len(m) - self.TAIL + 3) // 4 * 4
            m = m[off:]
        self.lp, _ = self.cml.predict(m, len(m))
        self.lp_off = off // 4
        a = OUT_PER_STEP * self.calls - DELAY
        self.calls += 1
        return self._rows(a)

    def flush(self):
        a = OUT_PER_STEP * self.calls - DELAY
        self.calls += 1
        return self._rows(a)


def load_step_model(kind="step", variant=None, precision="fp16", compute_units="CPU_AND_NE", mech=None):
    """variant None = 'faithful' for the step model (StreamStep's build), 'release' (the released
    multifunction window package) for the stand-in. mech: StepModel's state mechanism ('state' =
    ct.StateType, 'io' = explicit cache inputs/outputs); None = stream_step's default."""
    if kind == "step":
        from stream_step import StepModel
        kw = {} if mech is None else {"mech": mech}
        return StepModel(variant=variant or "faithful", precision=precision, compute_units=compute_units, **kw)
    if kind == "standin":
        from cmlrun import RELEASE, V1CoreML
        return WindowStandIn(V1CoreML(variant or RELEASE, precision, compute_units))
    raise ValueError(f"kind must be 'step' or 'standin', got {kind!r}")


# ---------------------------------------------------------------- decoder
class BeamIds:
    """V1's ctc-beam-decoder (dec_cbd.CbdDecoder, V1_DECODER) returning token ids: the same call
    CbdDecoder.decode makes, with the piece labels mapped back to ids before sentencepiece."""

    def __init__(self, dec=None):
        self.dec = dec or beam_decoder()
        if self.dec.units != "piece":
            raise ValueError("V1 decodes with piece units")

    def ids(self, logp):
        if len(logp) == 0:
            return []
        beams = self.dec.dec.decode_beams(np.asarray(logp, dtype=np.float32), **self.dec.kw)
        return [int(w[1:]) for w in beams[0][0].split()] if beams else []

    @staticmethod
    def words(ids):
        return norm_text(sp().decode(ids)).split()


# ---------------------------------------------------------------- recognizer
class StreamingRecognizer:
    def __init__(self, step_model, decoder="A", pause_s=0.6, max_segment_s=35.0, preroll_s=0.5, beam=None):
        if decoder != "A":
            raise ValueError("only mode A (V1 beam+LM, every word shown) is implemented")
        self.model = step_model
        self.beam = beam or BeamIds()
        self.pause = None if pause_s is None else int(round(pause_s / FRAME_S))
        self.max_seg = None if max_segment_s is None else int(round(max_segment_s / FRAME_S))
        self.preroll = int(round(preroll_s / FRAME_S))
        self.mel = IncrementalMel()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.model.reset()
        self.mel.reset()
        self.n = 0                         # samples received
        self.j = 0                         # step-model calls made
        self.pend = []                     # mel rows of the next step
        self.pend_mel_ms = 0.0
        self.level_buf = []                # samples since the previous step (for 'level')
        self.lp = np.zeros((1024, N_CLASSES), np.float32)   # log-probs of the stream, growing
        self.nf = 0                        # output frames received (stream frame index)
        self.nonblank = np.zeros(1024, bool)
        self.seg_start = 0                 # stream frame where the open segment starts
        self.segment = 0
        self.committed = []
        self.shown = []
        self.shown_ids = []
        self.finished = False

    # ------------------------------------------------------------------ API
    def feed(self, samples):
        if self.finished:
            raise RuntimeError("finish() was called; reset() before a new stream")
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        events, pos = [], 0
        while pos < len(x):
            take = min(step_ready_samples(self.j) - self.n, len(x) - pos)
            part = x[pos:pos + take]
            t0 = time.perf_counter()
            rows = self.mel.feed(part)
            self.pend_mel_ms += (time.perf_counter() - t0) * 1e3
            if len(rows):
                self.pend.append(rows)
            self.level_buf.append(part)
            pos += take
            self.n += take
            if self.n == step_ready_samples(self.j):
                events += self._step()
        return events

    def finish(self):
        """Last partial chunk (if any), one flush call, final decode; closes the stream."""
        if self.finished:
            return []
        t0 = time.perf_counter()
        self.pend.append(self.mel.finish())
        mel_ms = self.pend_mel_ms + (time.perf_counter() - t0) * 1e3
        V = self.mel.n_valid
        L = -(-V // OUT_PER_STEP)
        rows = np.concatenate(self.pend) if self.pend else np.zeros((0, N_FEAT), np.float32)
        model_ms, calls = 0.0, 0
        if len(rows):                      # 1..15 valid mel frames left
            x = np.zeros((MEL_PER_STEP, N_FEAT), np.float32)
            x[:len(rows)] = rows
            t0 = time.perf_counter()
            out = self.model.step(x, n_valid=len(rows))
            model_ms += (time.perf_counter() - t0) * 1e3
            self._accept(out, L)
            calls += 1
        if OUT_PER_STEP * self.j - DELAY < L:
            t0 = time.perf_counter()
            out = self.model.flush()
            model_ms += (time.perf_counter() - t0) * 1e3
            self._accept(out, L)
            calls += 1
        if self.nf != L:
            raise RuntimeError(f"stream produced {self.nf} output frames, expected ceil({V}/4) = {L}")
        level = rms_db(np.concatenate(self.level_buf)) if self.level_buf else -100.0
        self.pend, self.pend_mel_ms, self.level_buf = [], 0.0, []
        self.finished = True
        ev = self._close(self.nf, level, {"mel_ms": mel_ms, "model_ms": model_ms, "calls": calls})
        return [ev] if ev is not None else []

    @property
    def t_audio(self):
        return self.n / SR

    def words(self):
        return self.committed + self.shown

    # ------------------------------------------------------------------ one step
    def _accept(self, out, L=None):
        """Append the valid rows of one model call (frames [4j - 4, 4j), j = calls so far)."""
        a = OUT_PER_STEP * self.j - DELAY
        self.j += 1
        out = np.asarray(out, np.float32)
        lo = max(0, -a)
        hi = len(out) if L is None else min(len(out), L - a)
        if hi <= lo:
            return 0
        new = out[lo:hi]
        if a + lo != self.nf:
            raise RuntimeError(f"frame bookkeeping: got frame {a + lo}, expected {self.nf}")
        need = self.nf + len(new)
        if need > len(self.lp):
            cap = max(need, 2 * len(self.lp))
            self.lp = np.concatenate([self.lp, np.zeros((cap - len(self.lp), N_CLASSES), np.float32)])
            self.nonblank = np.concatenate([self.nonblank, np.zeros(cap - len(self.nonblank), bool)])
        self.lp[self.nf:need] = new
        self.nonblank[self.nf:need] = new.argmax(1) != BLANK
        self.nf = need
        return len(new)

    def _step(self):
        x = np.concatenate(self.pend)
        if len(x) != MEL_PER_STEP:
            raise RuntimeError(f"step {self.j}: {len(x)} mel frames, expected {MEL_PER_STEP}")
        step_idx = self.j
        level = rms_db(np.concatenate(self.level_buf))
        mel_ms = self.pend_mel_ms
        self.pend, self.pend_mel_ms, self.level_buf = [], 0.0, []
        t0 = time.perf_counter()
        out = self.model.step(x)
        model_ms = (time.perf_counter() - t0) * 1e3
        got = self._accept(out)
        detail = {"mel_ms": mel_ms, "model_ms": model_ms, "calls": 1, "step": step_idx}
        events = []
        if self.max_seg is not None and self.nf - self.seg_start > self.max_seg:
            ev = self._close(self._quiet_cut(), level, detail)
            if ev is not None:
                events.append(ev)
            detail = {"mel_ms": 0.0, "model_ms": 0.0, "calls": 0, "step": step_idx}
        if self.pause is not None:
            seg = self.nonblank[self.seg_start:self.nf]
            if not seg.any():                          # nothing spoken yet: keep only the preroll
                self.seg_start = max(self.seg_start, self.nf - self.preroll)
            elif len(seg) >= self.pause and not seg[-self.pause:].any():
                ev = self._close(self.nf, level, detail)
                if ev is not None:
                    return events + [ev]
        dec_ms = 0.0
        if got or events:
            t0 = time.perf_counter()
            self.shown_ids = self.beam.ids(self.lp[self.seg_start:self.nf])
            self.shown = BeamIds.words(self.shown_ids)
            dec_ms = (time.perf_counter() - t0) * 1e3
        detail.update(decode_ms=dec_ms, frames=self.nf - self.seg_start)
        events.append(self._update(level, detail))
        return events

    def _update(self, level, detail):
        return {"kind": "update", "t_audio": self.t_audio, "segment": self.segment,
                "committed": list(self.committed), "live": self.shown[:-1],
                "pending": self.shown[-1] if self.shown else None,
                "compute_ms": detail["mel_ms"] + detail["model_ms"] + detail["decode_ms"],
                "window": "step", "level": level, "detail": detail}

    def _close(self, cut, level, detail):
        """Final beam+LM decode of frames [seg_start, cut); the rest stays in the next segment.
        Returns None if the segment is empty and nothing was shown."""
        t0 = time.perf_counter()
        ids = self.beam.ids(self.lp[self.seg_start:cut])
        final = BeamIds.words(ids)
        dec_ms = (time.perf_counter() - t0) * 1e3
        start = self.seg_start
        self.seg_start = cut
        if not final and not self.shown and not self.finished:
            return None
        replaced = self.shown
        ev = {"kind": "segment_end", "segment": self.segment, "final": final, "replaced": replaced,
              "changed_words": edit_distance(replaced, final), "t_audio": self.t_audio,
              "compute_ms": detail["mel_ms"] + detail["model_ms"] + dec_ms, "window": "step", "level": level,
              "detail": {**detail, "decode_ms": dec_ms, "frames": cut - start, "start_frame": start,
                         "end_frame": cut, "ids": ids}}
        self.committed += final
        self.segment += 1
        self.shown, self.shown_ids = [], []
        return ev

    def _quiet_cut(self):
        """Cut frame for an over-long segment: middle of the longest blank run in its last 5 s."""
        end = self.seg_start + self.max_seg
        lo = max(self.seg_start + 1, end - int(round(5 / FRAME_S)))
        nb = self.nonblank[lo:end]
        best, best_len, run = None, 0, 0
        for i, v in enumerate(nb):
            run = 0 if v else run + 1
            if run > best_len:
                best_len, best = run, i
        if best is None:
            return end
        return lo + best - best_len // 2


# ---------------------------------------------------------------- smoke check
def main():
    import argparse
    import json
    from common import OUT, load_audio, load_list
    from wer import corpus_wer
    ap = argparse.ArgumentParser(description="Run StreamingRecognizer on FAST utterances; compare the final text "
                                             "with the FAST reference beam transcript")
    ap.add_argument("--model", choices=("step", "standin"), default="standin")
    ap.add_argument("--mech", default=None, help="StepModel mechanism: state | io (default: stream_step's)")
    ap.add_argument("--variant", default=None, help="default: faithful (step), release (standin windows)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=1600, help="samples per feed() call")
    ap.add_argument("--pause-s", type=float, default=0.0, help="0 = one segment per utterance")
    args = ap.parse_args()
    model = load_step_model(args.model, args.variant, mech=args.mech)
    rec = StreamingRecognizer(model, pause_s=args.pause_s or None, max_segment_s=None if not args.pause_s else 35.0)
    ref = {u["id"]: u for u in json.loads((OUT / "reference/fast/index.json").read_text())["utts"]}
    refs, hyps, same = [], [], 0
    for u in load_list("fast")[:args.n]:
        w = load_audio(u["audio"])
        rec.reset()
        evs = []
        for a in range(0, len(w), args.chunk):
            evs += rec.feed(w[a:a + args.chunk])
        evs += rec.finish()
        text = " ".join(rec.committed)
        same += text == ref[u["id"]]["beam"]
        refs.append(u["ref"])
        hyps.append(text)
        steps = [e for e in evs if e["kind"] == "update"]
        print(json.dumps({"id": u["id"], "steps": len(steps), "segments": rec.segment, "same_as_ref_beam":
                          text == ref[u["id"]]["beam"], "model_ms_mean": float(np.mean(
                              [e["detail"]["model_ms"] for e in steps])) if steps else None, "text": text}),
              flush=True)
    print(json.dumps({"model": args.model, "utts": len(hyps), "same_as_ref_beam": same,
                      "wer": corpus_wer(refs, hyps)[0]}))


if __name__ == "__main__":
    main()
