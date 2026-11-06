"""Nine configurations, held-out predictions, retrieval and guard stress tests."""
import csv
import json
from pathlib import Path

import numpy as np

from .data import policy_id
from .decision import decide, keyword, unsafe_fallback
from .learning import fit_predict
from .retrieval import BM25

SYSTEMS = ("always_approve", "keyword", "tfidf_lr", "bm25_tfidf_lr", "fixed_v1",
           "unscoped_rules", "exact_lookup", "scoped_evidence", "no_material_checks")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics(labels, decisions):
    labels = np.asarray(labels)
    decisions = np.asarray(decisions)
    approved, rejected = decisions == "approve", decisions == "reject"
    tp, tn = int(np.sum(approved & (labels == 1))), int(np.sum(rejected & (labels == 0)))
    fp, fn = int(np.sum(approved & (labels == 0))), int(np.sum(rejected & (labels == 1)))
    escalated = int(np.sum(decisions == "escalate"))
    pos, neg = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / pos if pos else 0.
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "escalated": escalated,
            "coverage": 1 - escalated / len(labels),
            # Binary accuracy is undefined when any requests are escalated.
            "accuracy": (tp + tn) / len(labels) if not escalated else None,
            "balanced_accuracy": .5 * (tp / pos + tn / neg) if not escalated else None,
            "false_approval_rate": fp / neg,
            "precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}


def bootstrap_accuracy(labels, decisions, groups):
    correct = np.asarray(decisions) == np.where(np.asarray(labels) == 1, "approve", "reject")
    groups = np.asarray(groups)
    counts = np.array([correct[groups == g].sum() for g in range(15)])
    sizes = np.array([(groups == g).sum() for g in range(15)])
    rng = np.random.default_rng(20260925)
    samples = rng.integers(0, 15, size=(1000, 15))
    accuracy = counts[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    return [float(x) for x in np.quantile(accuracy, [.025, .975])]


def evaluate(assets, targets, folds, policies, out):
    out = Path(out)
    labels = np.array([int(t["compliant"]) for t in targets])
    groups = np.array([int(f["template_group"]) for f in folds])
    fold_ids = np.array([int(f["test_fold"]) for f in folds])
    index = {p["doc_id"]: p for p in policies}
    bm25 = BM25(policies)
    # Rank once per request and reuse; gold IDs are consulted only for scoring.
    rankings = [bm25.rank(x["text"]) for x in assets]
    top = [policies[r[0][0]] for r in rankings]
    retrieval_rows = []
    for i, ranks in enumerate(rankings):
        gold = targets[i]["gold_policy_id"]
        position = next(j+1 for j, (k, _) in enumerate(ranks) if policies[k]["doc_id"] == gold)
        retrieval_rows.append({"request_id": assets[i]["request_id"], "gold_rank": position,
                               "top1_doc_id": top[i]["doc_id"], "top1_score": ranks[0][1]})
    write_csv(out / "retrieval.csv", retrieval_rows)
    retrieval_summary = {f"recall_at_{k}": sum(r["gold_rank"] <= k for r in retrieval_rows)/len(assets)
                         for k in (1, 3, 5)}
    learned, fitting = {}, []
    for system in ("tfidf_lr", "bm25_tfidf_lr"):
        texts = [x["text"] + (" POLICY_EVIDENCE " + top[i]["text"] if system == "bm25_tfidf_lr" else "")
                 for i, x in enumerate(assets)]
        scores = np.zeros(len(assets))
        for fold in range(5):
            train, test = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
            assert not set(groups[train]) & set(groups[test])
            probability, audit = fit_predict([texts[i] for i in train], labels[train], [texts[i] for i in test])
            scores[test] = probability
            fitting.append({"system": system, "test_fold": fold, **audit,
                            "train_groups": sorted(set(int(x) for x in groups[train])),
                            "test_groups": sorted(set(int(x) for x in groups[test]))})
        learned[system] = scores
    rows, summary, decisions_by_system = [], [], {}
    for system in SYSTEMS:
        decisions = []
        for i, x in enumerate(assets):
            evidence = ""
            probability = ""
            if system in learned:
                probability = float(learned[system][i])
                decision, reason = ("approve" if probability >= .5 else "reject"), "trained_text_classifier"
                if system == "bm25_tfidf_lr":
                    evidence = top[i]["doc_id"]
            elif system == "always_approve":
                decision, reason = "approve", "majority_baseline"
            elif system == "keyword":
                decision, reason = keyword(x)
            else:
                if system == "fixed_v1":
                    p = index[policy_id(x["family"], "v1")]
                elif system == "unscoped_rules":
                    p = top[i]
                elif system == "exact_lookup":
                    p = index.get(policy_id(x["family"], x["revision"]))
                else:
                    p = bm25.scoped(x)
                evidence = p["doc_id"] if p else ""
                decision, reason = decide(x, p, guard=system not in ("fixed_v1", "unscoped_rules"),
                                          skip_material=system == "no_material_checks")
            decisions.append(decision)
            rows.append({"request_id": x["request_id"], "system": system, "fold": int(fold_ids[i]),
                         "template_group": int(groups[i]), "true_compliant": int(labels[i]),
                         "decision": decision, "reason": reason, "evidence_id": evidence,
                         "probability_compliant": probability})
        m = metrics(labels, decisions)
        lo, hi = bootstrap_accuracy(labels, decisions, groups)
        summary.append({"system": system, **m, "accuracy_ci_low": lo, "accuracy_ci_high": hi})
        decisions_by_system[system] = decisions
    write_csv(out / "predictions.csv", rows)
    write_csv(out / "classification.csv", summary)
    write_json(out / "fold_training_audit.json", fitting)
    write_json(out / "retrieval_summary.json", retrieval_summary)
    removed = {policy_id(f, "v2") for f in ("titanium_alloy", "aluminum_alloy", "ceramic")}
    incomplete = BM25([p for p in policies if p["doc_id"] not in removed])
    stress_rows, stress_summary = [], []
    for scenario in ("complete_guard", "fixed_v1", "unscoped_rules", "no_material_checks",
                     "missing_guard", "missing_unsafe", "wrong_version_guard", "wrong_version_unsafe"):
        decisions = []
        for i, x in enumerate(assets):
            if scenario in ("fixed_v1", "unscoped_rules", "no_material_checks"):
                decision, reason = decisions_by_system[scenario][i], "see_predictions_csv"
            else:
                p = (incomplete.scoped(x) if scenario.startswith("missing") else
                     index[policy_id(x["family"], "v0")] if scenario.startswith("wrong_version") else bm25.scoped(x))
                decision, reason = unsafe_fallback(x, p) if scenario.endswith("unsafe") else decide(x, p)
            decisions.append(decision)
            stress_rows.append({"scenario": scenario, "request_id": x["request_id"],
                                "true_compliant": int(labels[i]), "decision": decision, "reason": reason})
        stress_summary.append({"scenario": scenario, **metrics(labels, decisions)})
    write_csv(out / "stress_predictions.csv", stress_rows)
    write_csv(out / "stress_summary.csv", stress_summary)
    return summary, stress_summary, retrieval_summary
