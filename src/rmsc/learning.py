"""Train-only sparse TF-IDF and NumPy logistic-regression optimizer."""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from .retrieval import tokens


def features(text):
    unigram = ["NUM" if t.isnumeric() else t for t in tokens(text)]
    return Counter(unigram + [a + " " + b for a, b in zip(unigram, unigram[1:])])


class Tfidf:
    def fit(self, texts):
        df = Counter(term for text in texts for term in features(text))
        ordered = sorted(df, key=lambda t: (-df[t], t))[:1800]
        self.vocab = {t: j for j, t in enumerate(ordered)}
        self.idf = np.array([np.log((1 + len(texts)) / (1 + df[t])) + 1 for t in ordered])
        return self

    def transform(self, texts):
        rows, cols, values = [], [], []
        for i, text in enumerate(texts):
            for t, count in features(text).items():
                if t in self.vocab:
                    j = self.vocab[t]
                    rows.append(i)
                    cols.append(j)
                    values.append((1 + np.log(count)) * self.idf[j])
        x = csr_matrix((values, (rows, cols)), shape=(len(texts), len(self.vocab)))
        norm = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        return x.multiply((1 / np.maximum(norm, 1e-15))[:, None]).tocsr()


def fit_predict(train_texts, y, test_texts):
    transform = Tfidf().fit(train_texts)
    x, test = transform.transform(train_texts), transform.transform(test_texts)
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.zeros(x.shape[1])
    intercept = 0.
    class_counts = np.bincount(y.astype(int), minlength=2)
    sample_weight = n / (2 * class_counts[y.astype(int)])
    for _ in range(250):
        score = x @ w + intercept
        probability = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
        residual = (probability - y) * sample_weight
        w -= np.asarray(x.T @ residual).ravel() / n + .002 * w
        intercept -= float(residual.mean())
    logits = test @ w + intercept
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    return probabilities, {"vocabulary_size": len(transform.vocab), "training_records": n,
                           "positive_training_records": int(y.sum())}
