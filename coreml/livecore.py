"""Live V1 transcription engine on Core ML: 16 kHz mono audio in, display events out.

    from livecore import LiveEngine
    eng = LiveEngine("A", cml=V1CoreML("release", "fp16", "CPU_AND_NE"))   # the released package
    for ev in eng.feed(samples): ...        # float32 16 kHz mono, any length, repeatedly
    for ev in eng.flush(): ...              # optional: process audio since the last 0.4 s boundary now
    for ev in eng.finish(): ...             # end of stream: closes the open segment
    eng.reset()                             # new stream (same model and decoders)
    other = LiveEngine("C", cml=eng.cml, decoders=eng.decoders)   # switch mode, reuse model + LM

Loop. Every step_s (0.4 s) of new audio the engine re-encodes the CURRENT segment, from its start to
now, with the smallest Core ML window that fits (common.mel + V1CoreML.predict; the model is causal
with a 160 ms lookahead, so re-encoding only refreshes the last frames and the segment edges), decodes
it and emits an 'update'. A segment ends after pause_s (0.6 s) of silence, or when it would exceed
max_segment_s (35 s, the largest window); its final text is V1 beam+LM (dec_cbd, V1_DECODER) on the
segment's log-probs and is emitted as 'segment_end', then appended to the committed words.

Modes.
  A  V1 beam+LM (ctc-beam-decoder, piece units, subword 6-gram, beam 32, alpha 0.8, beta 1) on the
     segment at every step; every word shown, the last one flagged as pending (still being spelled).
  C  greedy + LM fusion (dec_subword.SubwordBeamDecoder, beam 1, alpha 0.5, beta 1.5, same 6-gram)
     at every step; at each pause the segment's text is replaced by V1 beam+LM ('replaced' = the
     live words before the correction, 'changed_words' = word edit distance replaced -> final).
  The final text of a segment is the same in both modes (same VAD, same audio, same beam+LM).

Segmentation (energy VAD on the waveform, 10 ms frames of 160 samples):
  frame level  db = 10 log10(mean(x^2) + 1e-10) dBFS
  noise floor  adaptive, in dB, starts at the first frame's level (clamped at -90 dBFS); a frame
               below the floor pulls it down quickly (floor += 0.2 (db - floor)); otherwise it rises
               slowly, +3 dB per second, so it re-learns a louder background within seconds while the
               dips between words keep it low during speech
  speech frame db > max(floor + 12 dB, -70 dBFS), the floor as it was before this frame
  onset        a segment opens when >= 5 of the last 20 frames (50 ms in 200 ms) are speech; it
               starts 0.5 s (preroll_s) before the earliest of those speech frames, rounded down to the
               stream's 640-sample grid (so the segment's output frames fall on the same 40 ms grid as a
               whole-stream encode), never before the previous segment's end; while no segment is open
               nothing is encoded
  pause        the open segment ends at the first step boundary where the audio since the end of
               the last speech frame is >= pause_s (0.6 s); the segment includes that silence
  max length   if the segment would exceed max_segment_s it is cut at the quietest point (100 ms
               moving average of the frame level) in its last 5 s, on the 640-sample (4 mel frame)
               grid; the rest opens the next segment at once
Compute is measured inside the engine (time.perf_counter) per event: compute_ms covers the VAD, mel,
Core ML prediction and decoding done for that event; 'detail' breaks it down for latency studies.

Events (all word fields are lists of lowercase words):
  {'kind': 'update', 't_audio': s of stream audio consumed, 'segment': int (0-based),
   'committed': final words of all earlier segments, 'live': current segment's words except the last,
   'pending': its last word or None, 'compute_ms', 'window': '5s'|'10s'|'20s'|'35s' or None (no
   segment open: nothing encoded), 'level': RMS dBFS of this step's input audio (floor -100),
   'detail': {...}}
  {'kind': 'segment_end', 'segment', 'final': words, 'replaced': words shown before the correction,
   'changed_words': int, 't_audio', 'compute_ms', 'window', 'level', 'detail': {...}}
"""
import time

import numpy as np

from common import LM, SPM, SR, WINDOWS, beam_decoder, mel, norm_text
from wer import edit_distance

FRAME = 160                      # VAD frame, 10 ms
GRID = 640                       # 4 mel frames = 1 output frame (40 ms)
MAX_MEL = max(WINDOWS.values())
FUSION = dict(alpha=0.5, beta=1.5, beam=1)


def rms_db(x):
    if len(x) == 0:
        return -100.0
    return max(-100.0, float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-10)))


def encode(cml, wav):
    """mel (common.mel, truncation=False) -> smallest window that fits -> Core ML log-probs [T, 257];
    (log_probs, info: window, samples, mel_ms, model_ms)."""
    t0 = time.perf_counter()
    m, n_valid = mel(wav)
    m = m[:n_valid]
    if n_valid > MAX_MEL:                # cannot happen for segments <= 35 s; keep the newest audio
        m, n_valid = m[-MAX_MEL:], MAX_MEL
    t1 = time.perf_counter()
    window = cml.window_for(n_valid)
    lp, _ = cml.predict(m, n_valid, window)
    t2 = time.perf_counter()
    return lp, {"window": window, "samples": len(wav), "mel_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3}


def build_decoders(mode):
    """{'beam': V1 ctc-beam-decoder} (+ 'fusion' for mode C); both expose decode(logp) -> text."""
    dec = {"beam": beam_decoder()}
    if mode == "C":
        from dec_subword import SubwordBeamDecoder
        dec["fusion"] = SubwordBeamDecoder(256, str(SPM), lm=str(LM), **FUSION)
    return dec


class EnergyVAD:
    """The frame rule in the module docstring. step(frame) -> (is_speech, frame dBFS)."""

    def __init__(self, margin_db=12.0, abs_min_db=-70.0, rise_db_s=3.0, fall=0.2, clamp_db=-90.0):
        self.margin, self.abs_min, self.fall, self.clamp = margin_db, abs_min_db, fall, clamp_db
        self.rise = rise_db_s * FRAME / SR
        self.floor = None

    def step(self, frame):
        db = 10 * np.log10(np.mean(np.square(frame, dtype=np.float64)) + 1e-10)
        d = max(db, self.clamp)
        if self.floor is None:
            self.floor = d
        speech = db > max(self.floor + self.margin, self.abs_min)
        self.floor += self.fall * (d - self.floor) if d < self.floor else self.rise
        return bool(speech), float(db)


class LiveEngine:
    def __init__(self, mode="A", cml=None, step_s=0.4, pause_s=0.6, max_segment_s=35.0, preroll_s=0.5,
                 decoders=None, encoder=None):
        """cml: a loaded V1CoreML (shared between engines); decoders: reuse another engine's
        .decoders (must contain 'fusion' for mode C); encoder: optional callable(wav, key) ->
        (log_probs, info) replacing mel + predict (info: window, samples, mel_ms, model_ms)."""
        if mode not in ("A", "C"):
            raise ValueError(f"mode must be 'A' or 'C', got {mode!r}")
        if cml is None and encoder is None:
            raise ValueError("pass cml (V1CoreML) or encoder")
        self.mode, self.cml = mode, cml
        self.step = int(round(step_s * SR))
        self.pause = int(round(pause_s * SR))
        self.max_seg = int(round(max_segment_s * SR))
        self.preroll = int(round(preroll_s * SR))
        self.decoders = decoders if decoders is not None else build_decoders(mode)
        if mode == "C" and "fusion" not in self.decoders:
            self.decoders = {**self.decoders, **build_decoders("C")}
        self.live_decoder = "beam" if mode == "A" else "fusion"
        self.encoder = encoder or self._encode
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.buf = np.zeros(0, np.float32)
        self.buf_start = 0               # absolute sample index of buf[0]
        self.n = 0                       # samples received
        self.done = 0                    # samples processed (last step boundary)
        self.vad_pos = 0                 # next VAD frame start
        self.vad = EnergyVAD()
        self.recent = []                 # (frame start, is_speech) of the last 20 VAD frames
        self.levels = []                 # frame dB since buf_start (for the max-length cut)
        self.last_speech = None          # end sample of the last speech frame
        self.active = False
        self.seg_start = 0
        self.segment = 0
        self.committed = []
        self.shown = []                  # words currently shown for the open segment
        self._last = None                # (key, log_probs, info) of the latest encode

    @property
    def t_audio(self):
        return self.done / SR

    @property
    def last_speech_s(self):
        return None if self.last_speech is None else self.last_speech / SR

    def _audio(self, a, b):
        return self.buf[a - self.buf_start:b - self.buf_start]

    def _trim(self, keep_from):
        """Drop whole VAD frames before keep_from (and before any audio the VAD has not seen)."""
        k = (min(keep_from, self.vad_pos) - self.buf_start) // FRAME
        if k > 0:
            self.buf = self.buf[k * FRAME:]
            self.levels = self.levels[k:]
            self.buf_start += k * FRAME

    # ------------------------------------------------------------------ API
    def feed(self, samples):
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, x]) if len(self.buf) else x.copy()
        self.n += len(x)
        events = []
        while self.n - self.done >= self.step:
            events += self._step(self.done + self.step, final=False)
        return events

    def flush(self):
        """Process the audio received since the last step boundary as one step (it may close the
        segment on a pause) without forcing the segment to end."""
        return self._step(self.n, final=False) if self.n > self.done else []

    def finish(self):
        """Process the remaining audio and close the open segment."""
        return self._step(self.n, final=True)

    # ------------------------------------------------------------------ one step
    def _run_vad(self, end):
        while self.vad_pos + FRAME <= end:
            fr = self._audio(self.vad_pos, self.vad_pos + FRAME)
            sp, db = self.vad.step(fr)
            self.levels.append(db)
            self.recent.append((self.vad_pos, sp))
            if len(self.recent) > 20:
                self.recent.pop(0)
            if sp:
                self.last_speech = self.vad_pos + FRAME
            if not self.active and sum(s for _, s in self.recent) >= 5:
                onset = min(p for p, s in self.recent if s)
                self.active = True
                self.seg_start = max(self.seg_start, (onset - self.preroll) // GRID * GRID)
                self._last = None
            self.vad_pos += FRAME

    def _step(self, end, final):
        t0 = time.perf_counter()
        level = rms_db(self._audio(self.done, end))
        self._run_vad(end)
        self.done = end
        vad_ms = (time.perf_counter() - t0) * 1e3
        events = []
        if not self.active:
            self.seg_start = max(self.seg_start, (end - self.preroll) // GRID * GRID)
            self._trim(self.seg_start)
            events.append(self._update(None, level, vad_ms, {"vad_ms": vad_ms}))
            return events
        if end - self.seg_start > self.max_seg:
            cut = self._quiet_cut(end)
            events.append(self._close(cut, level, vad_ms))
            vad_ms = 0.0
            self.active = True               # the rest of the audio opens the next segment
            self.seg_start = cut
        if final or self.last_speech is None or end - self.last_speech >= self.pause:
            events.append(self._close(end, level, vad_ms))
            self._trim(self.seg_start)
            return events
        lp, info, enc_ms = self._encode_cached(self.seg_start, end)
        t1 = time.perf_counter()
        words = norm_text(self.decoders[self.live_decoder].decode(lp)).split()
        dec_ms = (time.perf_counter() - t1) * 1e3
        self.shown = words
        detail = {"vad_ms": vad_ms, **info, "decoder": self.live_decoder, "decode_ms": dec_ms,
                  "frames": len(lp)}
        events.append(self._update(info["window"], level, vad_ms + enc_ms + dec_ms, detail))
        return events

    def _update(self, window, level, compute_ms, detail):
        return {"kind": "update", "t_audio": self.done / SR, "segment": self.segment,
                "committed": list(self.committed), "live": self.shown[:-1],
                "pending": self.shown[-1] if self.shown else None, "compute_ms": compute_ms,
                "window": window, "level": level, "detail": detail}

    def _close(self, cut, level, vad_ms):
        lp, info, enc_ms = self._encode_cached(self.seg_start, cut)
        t1 = time.perf_counter()
        final = norm_text(self.decoders["beam"].decode(lp)).split() if len(lp) else []
        dec_ms = (time.perf_counter() - t1) * 1e3
        replaced = self.shown
        ev = {"kind": "segment_end", "segment": self.segment, "final": final, "replaced": replaced,
              "changed_words": edit_distance(replaced, final), "t_audio": self.done / SR,
              "compute_ms": vad_ms + enc_ms + dec_ms, "window": info["window"], "level": level,
              "detail": {"vad_ms": vad_ms, **info, "decoder": "beam", "decode_ms": dec_ms, "frames": len(lp),
                         "start_s": self.seg_start / SR, "end_s": cut / SR}}
        self.committed += final
        self.segment += 1
        self.shown = []
        self.active = False
        self.seg_start = cut
        self._last = None
        return ev

    def _quiet_cut(self, end):
        lo = self.seg_start + self.max_seg - 5 * SR
        hi = self.seg_start + self.max_seg
        lv = np.asarray(self.levels, np.float64)
        smooth = np.convolve(lv, np.ones(10) / 10, mode="same")
        f0 = max(0, (lo - self.buf_start) // FRAME)
        f1 = min(len(smooth), (hi - self.buf_start) // FRAME)
        if f1 <= f0:
            cut = hi
        else:
            cut = self.buf_start + (f0 + int(np.argmin(smooth[f0:f1]))) * FRAME + FRAME // 2
        cut = self.seg_start + ((cut - self.seg_start) // GRID) * GRID
        return int(min(max(cut, self.seg_start + GRID), hi, end))

    # ------------------------------------------------------------------ model
    def _encode_cached(self, a, b):
        """(log_probs, info, encode ms actually spent); the same audio is not encoded twice."""
        key = (a, b)
        if self._last is not None and self._last[0] == key:
            return self._last[1], {**self._last[2], "reused": True}, 0.0
        if b - a < FRAME * 2:
            lp, info = np.zeros((0, 257), np.float32), {"window": None, "samples": b - a, "mel_ms": 0.0,
                                                        "model_ms": 0.0}
        else:
            lp, info = self.encoder(self._audio(a, b), key)
        self._last = (key, lp, info)
        return lp, {**info, "reused": False}, info["mel_ms"] + info["model_ms"]

    def _encode(self, wav, key=None):
        return encode(self.cml, wav)
