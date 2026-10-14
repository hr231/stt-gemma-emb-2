"""Stream-lowercase an ARPA LM (gz or plain) and write its unigram list.

  python lower_arpa.py in.arpa.gz out.lower.arpa [--unigrams out.unigrams.txt]

Lowercases every n-gram line (headers \\data\\, ngram N=, \\N-grams:, \\end\\ are already lowercase;
<s>, </s>, <unk> stay as they are). Fails if lowercasing would merge two distinct unigrams
(duplicates would make the ARPA invalid). Unigram list: one word per line, without <s>, </s>, <unk>.
"""
import argparse
import gzip
import sys

SPECIAL = {"<s>", "</s>", "<unk>"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--unigrams")
    a = ap.parse_args()
    opener = gzip.open if a.src.endswith(".gz") else open
    section, n_lines, uni = None, 0, []
    with opener(a.src, "rt", encoding="utf-8") as f, open(a.dst, "w", encoding="utf-8") as w:
        for line in f:
            if line.startswith("\\"):
                section = line.strip()
                w.write(line)
                continue
            low = line.lower()
            w.write(low)
            n_lines += 1
            if section == "\\1-grams:" and low.strip():
                word = low.split("\t")[1].strip()
                if word not in SPECIAL:
                    uni.append(word)
    if len(set(uni)) != len(uni):
        sys.exit(f"ERROR: lowercasing merged {len(uni) - len(set(uni))} unigrams")
    if a.unigrams:
        with open(a.unigrams, "w", encoding="utf-8") as u:
            u.write("\n".join(uni) + "\n")
    print(f"{a.dst}: {n_lines:,} lines, {len(uni):,} unigrams (excl. specials)")


if __name__ == "__main__":
    main()
