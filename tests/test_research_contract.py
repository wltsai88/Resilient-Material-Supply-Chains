"""Checks for scientific/data contracts, not manuscript-number fitting."""
import copy
import csv
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rmsc.data import generate_assets, generate_policies, independent_audit
from rmsc.decision import decide, unsafe_fallback, violations
from rmsc.evaluation import metrics
from rmsc.learning import Tfidf
from rmsc.retrieval import BM25
from rmsc.simulation import cost_scenarios, negotiate


class ResearchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets, cls.targets, cls.folds = generate_assets()
        cls.policies = generate_policies()
        cls.index = {p["doc_id"]: p for p in cls.policies}

    def test_population_and_unique_inputs(self):
        self.assertEqual(len(self.assets), 3255)
        self.assertEqual(set(Counter(x["family"] for x in self.assets).values()), {465})
        self.assertEqual(sum(t["compliant"] for t in self.targets), 2604)
        negatives = Counter(t["violation"] for t in self.targets if not t["compliant"])
        self.assertEqual(len(negatives), 7)
        self.assertEqual(set(negatives.values()), {93})
        self.assertEqual(len({t["input_sha256"] for t in self.targets}), 3255)

    def test_independent_audit_and_single_violation(self):
        for x, t in zip(self.assets, self.targets):
            faults = independent_audit(x)
            self.assertEqual(faults, [] if t["compliant"] else [t["violation"]])
            self.assertEqual(faults, violations(x, self.index[t["gold_policy_id"]]))

    def test_seed_is_deterministic(self):
        digest = lambda items: hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest(generate_assets()), digest((self.assets, self.targets, self.folds)))

    def test_no_targets_in_operational_records(self):
        forbidden = {"compliant", "label", "violation", "gold_policy_id", "template_group", "test_fold"}
        for x in self.assets:
            self.assertFalse(forbidden.intersection(x))
            self.assertNotIn("release:", x["text"])

    def test_folds_hold_out_complete_template_groups(self):
        self.assertEqual(set(Counter(f["template_group"] for f in self.folds).values()), {217})
        for fold in range(5):
            test = [f for f in self.folds if f["test_fold"] == fold]
            train = [f for f in self.folds if f["test_fold"] != fold]
            self.assertEqual((len(train), len(test)), (2604, 651))
            self.assertFalse({f["template_group"] for f in train} & {f["template_group"] for f in test})

    def test_policy_counts_and_guarded_decisions(self):
        self.assertEqual(Counter(p["type"] for p in self.policies), {"release": 28, "context": 56})
        bm25 = BM25(self.policies)
        for x, t in zip(self.assets, self.targets):
            p = bm25.scoped(x)
            self.assertEqual(p["doc_id"], t["gold_policy_id"])
            self.assertEqual(decide(x, p)[0], "approve" if t["compliant"] else "reject")

    def test_evidence_guard_and_explicit_unsafe_fallback(self):
        x = self.assets[0]
        p = copy.deepcopy(self.index[self.targets[0]["gold_policy_id"]])
        p["revision"] = "v0"
        self.assertEqual(decide(x, p)[0], "escalate")
        self.assertEqual(decide(x, None)[0], "escalate")
        self.assertEqual(unsafe_fallback(x, p)[0], "approve")
        self.assertEqual(unsafe_fallback(x, None)[0], "approve")
        p["revision"] = x["revision"]
        p["type"] = "context"
        self.assertEqual(decide(x, p)[0], "escalate")

    def test_material_ablation_exposes_both_categories(self):
        false_approvals = sum(decide(x, self.index[t["gold_policy_id"]], skip_material=True)[0] == "approve"
                              and not t["compliant"] for x, t in zip(self.assets, self.targets))
        self.assertEqual(false_approvals, 186)

    def test_train_only_vocabulary_and_normalization(self):
        tfidf = Tfidf().fit(["train token 123", "train other 345"])
        self.assertNotIn("heldout", tfidf.vocab)
        transformed = tfidf.transform(["heldout train 987", "heldout"])
        self.assertAlmostEqual(float(transformed[0].multiply(transformed[0]).sum()), 1.)
        self.assertEqual(transformed[1].nnz, 0)
        self.assertIn("NUM", tfidf.vocab)

    def test_bm25_tie_and_empty_scope(self):
        docs = [{"doc_id": i, "text": "identical token"} for i in ("b", "a")]
        bm25 = BM25(docs)
        self.assertEqual(bm25.documents[bm25.rank("identical")[0][0]]["doc_id"], "a")
        self.assertEqual(bm25.rank("identical"), bm25.rank("identical identical"))
        self.assertIsNone(BM25([]).scoped(self.assets[0]))

    def test_cost_model_and_bargaining_bounds(self):
        q20 = next(r for r in cost_scenarios() if r["invalid_ratio"] == .2)
        self.assertEqual(q20["baseline_cost"], 181681080)
        self.assertEqual(q20["prevalidated_cost"], 159992364)
        self.assertLess(cost_scenarios()[0]["reduction_fraction"], 0)
        self.assertEqual(negotiate(1., .1, .4, .9, .9), (1, .7))
        self.assertEqual(negotiate(.1, .7, .9, .9, .9), (None, None))

    def test_escalation_is_not_correct_rejection(self):
        m = metrics([1, 0], ["escalate", "reject"])
        self.assertIsNone(m["accuracy"])
        self.assertEqual(m["coverage"], .5)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fn"], 0)


if __name__ == "__main__":
    unittest.main()
