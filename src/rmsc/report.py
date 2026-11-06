"""Figures and explicit comparison with reported claims, never result fitting."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import write_csv

PAPER_CONFUSION = {
    "always_approve": (2604, 0, 651, 0), "keyword": (2604, 30, 621, 0),
    "tfidf_lr": (2307, 279, 372, 297), "bm25_tfidf_lr": (2389, 294, 357, 215),
    "unscoped_rules": (2575, 575, 76, 29), "exact_lookup": (2604, 651, 0, 0),
    "scoped_evidence": (2604, 651, 0, 0), "fixed_v1": (1336, 612, 39, 1268),
    "no_material_checks": (2604, 465, 186, 0)}
PAPER_STRESS = {"complete_guard": (0, 0), "fixed_v1": (39, 0), "unscoped_rules": (76, 0),
                "no_material_checks": (186, 0), "missing_guard": (0, 706),
                "missing_unsafe": (158, 0), "wrong_version_guard": (0, 3255),
                "wrong_version_unsafe": (651, 0)}


def make_report(out, main, stress, retrieval, costs, bargaining):
    comparison = []
    for row in main:
        for key, expected in zip(("tp", "tn", "fp", "fn"), PAPER_CONFUSION[row["system"]]):
            comparison.append({"experiment": row["system"], "metric": key,
                               "paper_claim": expected, "computed": row[key],
                               "delta": row[key] - expected, "matches": row[key] == expected})
    for row in stress:
        for key, expected in zip(("fp", "escalated"), PAPER_STRESS[row["scenario"]]):
            comparison.append({"experiment": row["scenario"], "metric": key,
                               "paper_claim": expected, "computed": row[key],
                               "delta": row[key] - expected, "matches": row[key] == expected})
    for key, expected_count in (("recall_at_1", 3024), ("recall_at_3", 3249), ("recall_at_5", 3255)):
        actual_count = round(retrieval[key] * 3255)
        comparison.append({"experiment": "unscoped_retrieval", "metric": key + "_count",
                           "paper_claim": expected_count, "computed": actual_count,
                           "delta": actual_count - expected_count, "matches": actual_count == expected_count})
    write_csv(out / "paper_comparison.csv", comparison)
    lines = ["# Computed benchmark results", "",
             "This is a new implementation of the attached manuscript specification. "
             "Its synthetic data are newly generated; they are not recovered original observations. "
             "Historical Table 7/8 counts are comparison targets only and never inference inputs.", "",
             "## Classification", "", "|System|TP|TN|FP|FN|Accuracy|Balanced accuracy|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in main:
        lines.append(f"|{r['system']}|{r['tp']}|{r['tn']}|{r['fp']}|{r['fn']}|{r['accuracy']:.2%}|{r['balanced_accuracy']:.2%}|")
    lines += ["", "## Evidence stress tests", "", "|Scenario|False approvals|Escalated|Coverage|",
              "|---|---:|---:|---:|"]
    for r in stress:
        lines.append(f"|{r['scenario']}|{r['fp']}|{r['escalated']}|{r['coverage']:.2%}|")
    failed = sum(not row["matches"] for row in comparison)
    lines += ["", "## Comparison with manuscript", "",
              f"{failed} of {len(comparison)} individual count checks differ. "
              "See paper_comparison.csv for every computed value and difference. "
              "Matching guard invariants do not establish recovery of the original random dataset. "
              "Different templates, lexical policy text, allocation, and RNG draw order can change retrieval and learned results.",
              "", "## Separate numerical models", ""]
    q20 = next(r for r in costs if r["invalid_ratio"] == .2)
    lines += [f"At 20% invalid submissions, baseline = {q20['baseline_cost']:,} and prevalidated cost = "
              f"{q20['prevalidated_cost']:,} gas-equivalent units; reduction = {q20['reduction_fraction']:.4%}.", "",
              "|Concurrency|Successful-agreement latency mean (seconds)|SD of run means|Agreement rate|",
              "|---|---:|---:|---:|"]
    for r in bargaining:
        lines.append(f"|{r['concurrency']}|{r['conditional_latency_mean']:.4f}|{r['latency_run_mean_sd']:.4f}|{r['agreement_rate_mean']:.2%}|")
    lines += ["", "These models are not measured ledger gas, queueing performance, industrial trials, or LLM inference.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([r["invalid_ratio"]*100 for r in costs], [r["reduction_fraction"]*100 for r in costs], marker="o")
    ax.axhline(0, color="grey", lw=.8)
    ax.set(xlabel="Invalid submissions (%)", ylabel="Modeled cost reduction (%)", title="Gas-equivalent cost model")
    fig.tight_layout(); fig.savefig(out / "figure3_cost.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    loads = [r["concurrency"] for r in bargaining]
    for ax, key, error, title in ((axes[0], "conditional_latency_mean", "latency_run_mean_sd", "Conditional latency (seconds)"),
                                  (axes[1], "agreement_rate_mean", "agreement_rate_run_sd", "Agreement rate")):
        ax.errorbar(loads, [r[key] for r in bargaining], yerr=[r[error] for r in bargaining], marker="o", capsize=3)
        ax.set(xscale="log", xlabel="Concurrency (timing parameter)", ylabel=title)
    fig.suptitle("Bounded bargaining model: 10 runs per load")
    fig.tight_layout(); fig.savefig(out / "figure4_bargaining.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ys = [r["accuracy"]*100 for r in main]
    lower = [max(0, r["accuracy"]-r["accuracy_ci_low"])*100 for r in main]
    upper = [max(0, r["accuracy_ci_high"]-r["accuracy"])*100 for r in main]
    ax.barh([r["system"] for r in main], ys, xerr=[lower, upper], color="#326a90", capsize=3)
    ax.set(xlim=(0, 105), xlabel="Accuracy (%)", title="New synthetic corpus with group-bootstrap intervals")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out / "figure5_classification.png"); plt.close(fig)
