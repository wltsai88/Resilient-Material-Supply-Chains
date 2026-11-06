This repository creates 3,255 new synthetic material requests and 84 policy documents, evaluates nine configurations, logs evidence and decisions, and runs separate cost and bargaining models.
Reconstruction status: the manuscript was supplied, but its original `run_benchmark.py`, exact policy/template text, original generated records, and random draw sequence were not supplied. This repository is a newly written implementation of the specification, not a recovered copy of the original experiment. Some manuscript counts differ from the computed outputs; all differences are exposed in `artifacts/results/paper_comparison.csv`. See `docs/PROVENANCE.md` before citing results.
Quick start
Python 3.10 or newer:
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python run_benchmark.py
python -m unittest discover -s tests -v
```
An existing `artifacts` directory is refreshed by a full run. To retain archived results, use a new output location:
```bash
python run_benchmark.py --output artifacts_new_run
python run_benchmark.py --generate-only --output generated_only
```
No API key, GPU, LLM service, blockchain node, or licensed policy corpus is needed. The stored outputs include all individual predictions and simulation outcomes. `requirements-lock.txt` records the versions used to generate the archived results; `requirements.txt` gives supported dependency ranges.
Where the 3,255 records come from
Item	Generation rule
Population	Seven material families × 465 requests = 3,255
Labels	2,604 compliant; 651 with exactly one violation
Negative categories	Seven categories × 93 = 651
Families	Titanium alloy, nickel superalloy, aluminum alloy, stainless steel, ceramic, polymer, composite
Policy corpus	Seven families × four revisions × (one release policy + two notes) = 84
Request revisions	v1 or v2, using zero-based policy revision indices v0–v3
Randomness	NumPy PCG64, seed 20260924
Composition	Dirichlet(2, 3, 4), six decimals, last component adjusted to sum to one
Text split	15 complete template groups × 217; five folds hold out three groups each
Independent check	A second predicate implementation audits every generated label
Process bounds, quota, territory, and redistribution rules follow manuscript §4.1. Every rule and newly chosen detail is documented in `docs/PROVENANCE.md`. These records are synthetic, and the policy values are fictional normalized examples, not usable manufacturing specifications.
Experiments
Always approve and restricted keyword baselines.
TF-IDF/logistic regression and BM25 top-1 policy plus TF-IDF/logistic regression, fitted on training folds only.
Fixed-v1 rules, unscoped BM25 plus rules, direct policy lookup, and scoped evidence checking.
Ablation without process/composition predicates.
Missing-policy and wrong-version stress tests, including explicitly unsafe approval comparators.
Analytical gas-equivalent cost model and 20-round bargaining simulation, with 10 runs at each of six load parameters.
BM25 uses positive IDF, k1=1.5, b=0.75, binary query-term occurrence, and document-ID tie-breaking. The two learned systems use a train-only 1,800-feature unigram/bigram vocabulary and class-balanced 250-step logistic optimization. There is no generative LLM or measured ledger deployment in this implementation.
Labels, assigned anomalies, gold policy IDs, and fold identifiers are stored separately from operational input records. A positive label means compliance; false approvals are approvals of noncompliant requests. Escalation is a third outcome, with coverage reported separately.
Outputs and actual results
Start with `artifacts/results/REPORT.md`. The report contains the actual computed results for this implementation and explains the comparison limits.
Path	Contents
`artifacts/data/requests.jsonl`	3,255 operational inputs, without labels or gold policy IDs
`artifacts/data/policies.jsonl`	84 fictional documents and explicit structured rules
`artifacts/data/targets.csv`	Labels, single-violation categories, gold evidence IDs, input hashes
`artifacts/data/folds.csv`	Template group and held-out test fold for each request
`artifacts/results/predictions.csv`	29,295 out-of-fold or fixed-system decisions with reasons and evidence
`artifacts/results/stress_predictions.csv`	26,040 individual stress-test decisions
`artifacts/results/retrieval.csv`	Per-request gold ranks and top-1 evidence
`artifacts/results/fold_training_audit.json`	Train/test group membership and fitted vocabulary sizes
`artifacts/results/bargaining_pairs.csv`	195,300 independently sampled pair outcomes and parameters
`artifacts/results/paper_comparison.csv`	Reported manuscript claims compared with computed counts
`artifacts/results/figure3_cost.png` … `figure5_classification.png`	Figures from computed outputs
`artifacts/manifest.json`	Seed, RNG, package versions, generation status, and counts
`artifacts/checksums.sha256`	SHA-256 for generated data/results
The scoped and direct-lookup systems both reach 100% on these generated deterministic rules; removing material checks exposes 186 false approvals. At 20% invalid submissions the cost equation yields 11.9378% savings. Template-sensitive and missing-evidence counts differ from the manuscript; inspect the comparison file rather than attributing historical table values to this run. Artificial agreement with those values is not enforced.
Source structure
Module	Responsibility
`src/rmsc/data.py`	Policies, synthetic requests, templates, independent label audit
`src/rmsc/retrieval.py`	BM25 and scope filtering
`src/rmsc/decision.py`	Predicates, evidence guard, decisions and reasons
`src/rmsc/learning.py`	Train-only TF-IDF and logistic optimizer
`src/rmsc/evaluation.py`	Folds, baselines, metrics, stress tests, bootstrap intervals
`src/rmsc/simulation.py`	Cost model and bargaining
`src/rmsc/report.py`	Figures and manuscript comparison
`tests/test_research_contract.py`	Data, leakage, evidence, metric and model checks
Chinese usage guide: `README_zh-TW.md`.
GitHub publication
The requested account is `wltsai88`; the intended repository slug is `Resilient-Material-Supply-Chains` (spaces replaced with hyphens). From an authenticated GitHub CLI environment, `bash scripts/publish_github.sh` checks the account, creates a private repository, and pushes this source and its generated outputs. It stops if the target already exists. This instruction is not evidence that the remote repository has been created.
No license is selected in this initial research package; the copyright holder can add a suitable code/data license before public release. The uploaded manuscript is not included in the repository; its SHA-256 and the relevant specification are recorded in `docs/SOURCE.json`.
