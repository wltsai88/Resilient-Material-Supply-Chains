import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy

from .data import SEED, generate_assets, generate_policies
from .evaluation import evaluate, write_csv, write_json
from .report import make_report
from .simulation import bargaining, cost_scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate 3,255 synthetic material requests")
    parser.add_argument("--output", default="artifacts", help="Output directory; default: artifacts")
    parser.add_argument("--generate-only", action="store_true", help="Generate/audit inputs without running models")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    data, results = root / "data", root / "results"
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    assets, targets, folds = generate_assets()
    policies = generate_policies()
    for name, rows in (("requests.jsonl", assets), ("policies.jsonl", policies)):
        with (data / name).open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
    write_csv(data / "targets.csv", targets)
    write_csv(data / "folds.csv", folds)
    manifest = {"implementation_version": "0.1.0", "status": "new_specification_based_reconstruction",
                "seed": SEED, "rng": "numpy.random.Generator(PCG64)",
                "requests": len(assets), "policies": len(policies), "template_groups": 15,
                "families": dict(Counter(x["family"] for x in assets)),
                "labels": dict(Counter(t["violation"] for t in targets)),
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    write_json(root / "manifest.json", manifest)
    print(f"Generated and audited {len(assets)} requests and {len(policies)} fictional policies", flush=True)
    if not args.generate_only:
        main_results, stress, retrieval = evaluate(assets, targets, folds, policies, results)
        print("Completed nine configurations, five held-out folds, retrieval and evidence stress tests", flush=True)
        costs = cost_scenarios()
        write_csv(results / "cost_scenarios.csv", costs)
        simulations = bargaining(results)
        make_report(results, main_results, stress, retrieval, costs, simulations)
        print("Completed cost and bargaining models; see results/REPORT.md", flush=True)
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "checksums.sha256"}
    (root / "checksums.sha256").write_text("".join(f"{digest}  {path}\n" for path, digest in checksums.items()))
    print(root, flush=True)


if __name__ == "__main__":
    main()
