"""Chunked CTC decoding with the search state kept between chunks (streaming study).

  StreamingBeam(dec_pyctc.PyctcDecoder(..., units="piece", fix_break=False))
      pyctcdecode 0.5.0's beam search, the algorithm ctc-beam-decoder reproduces (V1's decoder has no
      incremental API), split into accept(chunk) / partial_ids() / final_ids(). accept() is the
      per-frame body of BeamSearchDecoderCTC._decode_logits, final_ids() its final scoring, copied
      from pyctcdecode 0.5.0 (Apache-2.0) with the loop state (beams, LM caches, break flag, frame
      index) kept on the object.
  StreamingSubword(dec_subword.SubwordBeamDecoder)   dec_subword's start/step/best_ids/final_ids.
  StreamingGreedy(blank)                              argmax + CTC collapse.

partial_ids(): token ids of the current best hypothesis, without end-of-sentence scoring.
final_ids(): the result after the last chunk; equals decoding the whole utterance at once.
ctc_align_ends(): Viterbi CTC alignment of a token sequence -> exclusive end frame of each token.
"""
import importlib.metadata

import numpy as np


class StreamingGreedy:
    def __init__(self, blank):
        self.blank = blank
        self.reset()

    def reset(self):
        self.ids, self.prev = [], -1

    def accept(self, logp):
        for k in np.asarray(logp).argmax(1).tolist():
            if k != self.prev and k != self.blank:
                self.ids.append(k)
            self.prev = k

    def partial_ids(self):
        return list(self.ids)

    def final_ids(self):
        return list(self.ids)


class StreamingSubword:
    def __init__(self, dec):
        self.dec = dec
        self.reset()

    def reset(self):
        self.beams = self.dec.start()

    def accept(self, logp):
        self.beams = self.dec.step(self.beams, logp)

    def partial_ids(self):
        return self.dec.best_ids(self.beams)

    def final_ids(self):
        return self.dec.final_ids(self.beams)


class StreamingBeam:
    def __init__(self, wrapper):
        import pyctcdecode.decoder as pd
        if importlib.metadata.version("pyctcdecode") != "0.5.0":
            raise RuntimeError("StreamingBeam copies pyctcdecode 0.5.0's search loop; re-check it")
        if wrapper.units != "piece":
            raise ValueError("units='piece' only (labels \u2581t<id>)")
        if type(wrapper.dec) is not pd.BeamSearchDecoderCTC:
            raise ValueError("needs stock pyctcdecode (fix_break=False), as ctc-beam-decoder reproduces")
        self.pd, self.w, self.dec = pd, wrapper, wrapper.dec
        self.hot = pd.HotwordScorer.build_scorer(None, weight=pd.DEFAULT_HOTWORD_WEIGHT)
        self.reset()

    def reset(self):
        lm = self.dec._language_model
        self.cached = {("", False): (0.0, 0.0, lm.get_start_state() if lm is not None else None)}
        self.cached_p = {}
        self.beams = [self.pd.EMPTY_START_BEAM]
        self.force_next_break = False
        self.t = 0

    def accept(self, logp):
        pd, dec = self.pd, self.dec
        # decode_beams' input conversion (log-probs in: log_softmax again, clip at 1e-15), per row
        logits = np.clip(pd._log_softmax(np.asarray(logp, dtype=np.float32), axis=1), np.log(pd.MIN_TOKEN_CLIP_P), 0)
        beam_width, beam_prune_logp = self.w.beam, self.w.beam_prune_logp
        token_min_logp, prune_history = self.w.token_min_logp, self.w.prune_history
        language_model = dec._language_model
        beams, force_next_break = self.beams, self.force_next_break
        for local_idx, logit_col in enumerate(logits):
            frame_idx = self.t + local_idx
            max_idx = logit_col.argmax()
            idx_list = set(np.where(logit_col >= token_min_logp)[0]) | {max_idx}
            new_beams = []
            for idx_char in idx_list:
                p_char = logit_col[idx_char]
                char = dec._idx2vocab[idx_char]
                for (text, next_word, word_part, last_char, text_frames, part_frames, logit_score) in beams:
                    if char == "" or last_char == char:
                        if char == "":
                            new_end_frame = part_frames[0]
                        else:
                            new_end_frame = frame_idx + 1
                        new_part_frames = part_frames if char == "" else (part_frames[0], new_end_frame)
                        new_beams.append((text, next_word, word_part, char, text_frames, new_part_frames,
                                          logit_score + p_char))
                    elif dec._is_bpe and (char[:1] == pd.BPE_TOKEN or force_next_break):
                        force_next_break = False
                        clean_char = char
                        if char[:1] == pd.BPE_TOKEN:
                            clean_char = clean_char[1:]
                        if char[-1:] == pd.BPE_TOKEN:
                            clean_char = clean_char[:-1]
                            force_next_break = True
                        new_frame_list = text_frames if word_part == "" else text_frames + [part_frames]
                        new_beams.append((text, word_part, clean_char, char, new_frame_list,
                                          (frame_idx, frame_idx + 1), logit_score + p_char))
                    elif not dec._is_bpe and char == " ":
                        new_frame_list = text_frames if word_part == "" else text_frames + [part_frames]
                        new_beams.append((text, word_part, "", char, new_frame_list, pd.NULL_FRAMES,
                                          logit_score + p_char))
                    else:
                        new_part_frames = ((frame_idx, frame_idx + 1) if part_frames[0] < 0
                                           else (part_frames[0], frame_idx + 1))
                        new_beams.append((text, next_word, word_part + char, char, text_frames, new_part_frames,
                                          logit_score + p_char))
            new_beams = pd._merge_beams(new_beams)
            scored_beams = dec._get_lm_beams(new_beams, self.hot, self.cached, self.cached_p)
            max_score = max([b[-1] for b in scored_beams])
            scored_beams = [b for b in scored_beams if b[-1] >= max_score + beam_prune_logp]
            trimmed_beams = pd._sort_and_trim_beams(scored_beams, beam_width)
            if prune_history:
                lm_order = 1 if language_model is None else language_model.order
                beams = pd._prune_history(trimmed_beams, lm_order=lm_order)
            else:
                beams = [b[:-1] for b in trimmed_beams]
        self.beams, self.force_next_break = beams, force_next_break
        self.t += len(logits)

    def partial_ids(self):
        text, _, word_part = self.beams[0][:3]
        return [int(w[1:]) for w in text.split() + ([word_part] if word_part else [])]

    def final_ids(self):
        pd, dec = self.pd, self.dec
        new_beams = []
        for text, _, word_part, _, frame_list, frames, logit_score in self.beams:
            new_token_times = frame_list if word_part == "" else frame_list + [frames]
            new_beams.append((text, word_part, "", None, new_token_times, (-1, -1), logit_score))
        new_beams = pd._merge_beams(new_beams)
        scored_beams = dec._get_lm_beams(new_beams, self.hot, self.cached, self.cached_p, is_eos=True)
        max_score = max([b[-1] for b in scored_beams])
        scored_beams = [b for b in scored_beams if b[-1] >= max_score + self.w.beam_prune_logp]
        trimmed_beams = pd._sort_and_trim_beams(scored_beams, self.w.beam)
        text = pd._normalize_whitespace(trimmed_beams[0][0])
        return [int(w[1:]) for w in text.split()]


def ctc_align_ends(logp, ids, blank):
    """Viterbi CTC alignment of token ids to log-probs [T, C]: exclusive end frame of each token,
    or None if the sequence does not fit the frames."""
    T, L = len(logp), len(ids)
    if L == 0:
        return []
    ext = np.full(2 * L + 1, blank)
    ext[1::2] = ids
    S = len(ext)
    skip = np.zeros(S, bool)                      # token state s may follow token state s - 2
    skip[3::2] = np.asarray(ids[1:]) != np.asarray(ids[:-1])
    lp = np.asarray(logp, dtype=np.float64)[:, ext]
    neg = -1e30
    a = np.full(S, neg)
    a[0] = lp[0, 0]
    if S > 1:
        a[1] = lp[0, 1]
    bp = np.zeros((T, S), np.int8)
    cols = np.arange(S)
    for t in range(1, T):
        c1 = np.concatenate(([neg], a[:-1]))
        c2 = np.where(skip, np.concatenate(([neg, neg], a[:-2])), neg)
        stack = np.stack([a, c1, c2])
        k = stack.argmax(0)
        a = stack[k, cols] + lp[t]
        bp[t] = k
    s = S - 1 if S == 1 or a[S - 1] >= a[S - 2] else S - 2
    if a[s] < -1e29:
        return None
    ends = [0] * L
    for t in range(T - 1, -1, -1):
        if s % 2 == 1 and ends[s // 2] == 0:
            ends[s // 2] = t + 1
        s -= int(bp[t, s])
    return ends
