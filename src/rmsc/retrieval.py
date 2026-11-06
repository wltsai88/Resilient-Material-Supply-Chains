"""BM25 with positive IDF, binary query-term occurrence, deterministic ties."""
import re
from collections import Counter

import numpy as np


def tokens(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=.75):
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.counts = [Counter(tokens(d["text"])) for d in self.documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts], float)
        self.avg_length = float(self.lengths.mean()) if len(self.lengths) else 1.0
        df = Counter(term for c in self.counts for term in c)
        n = len(self.documents)
        self.idf = {t: np.log1p((n - count + .5) / (count + .5)) for t, count in df.items()}

    def rank(self, text, candidates=None):
        candidates = list(range(len(self.documents))) if candidates is None else list(candidates)
        scores = []
        for i in candidates:
            score = 0.
            normalizer = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            for term in set(tokens(text)):
                count = self.counts[i].get(term, 0)
                if count:
                    score += self.idf[term] * count * (self.k1 + 1) / (count + normalizer)
            scores.append((i, float(score)))
        return sorted(scores, key=lambda pair: (-pair[1], self.documents[pair[0]]["doc_id"]))

    def scoped(self, x):
        candidates = [i for i, d in enumerate(self.documents)
                      if d["type"] == "release" and d["family"] == x["family"]
                      and d["revision"] == x["revision"]]
        ranked = self.rank(x["text"], candidates)
        return self.documents[ranked[0][0]] if ranked else None
