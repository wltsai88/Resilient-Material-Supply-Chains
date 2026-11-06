"""Synthetic inputs and fictional policies reconstructed from manuscript §4.1.

Targets and gold policy IDs are kept outside operational input records.
Underspecified choices are documented in docs/PROVENANCE.md.
"""
import hashlib
import json
from collections import Counter

import numpy as np

FAMILIES = ("titanium_alloy", "nickel_superalloy", "aluminum_alloy",
            "stainless_steel", "ceramic", "polymer", "composite")
VIOLATIONS = ("safety", "quota", "territory", "process", "composition",
              "provenance", "redistribution")
SEED = 20260924


def policy_rules(f, v):
    """v is zero-based: v0/v1/v2/v3. Only v1 and v2 are requested."""
    return {"process_min": 0.12 + .035 * f + .035 * v,
            "process_max": .90 - .02 * f - .025 * v,
            "quota": 600 - 100 * v - 20 * f,
            "territory": "zone_a" if v % 2 else "zone_b",
            "redistribution_allowed": bool((f + v) % 2)}


def policy_id(family, revision):
    return f"release:{family}:{revision}"


def generate_policies():
    policies = []
    for f, family in enumerate(FAMILIES):
        for v in range(4):
            rules = policy_rules(f, v)
            for kind in ("release", "context_1", "context_2"):
                doc_id = f"{kind}:{family}:v{v}"
                text = (f"Material {family} policy v{v}. Production quota {rules['quota']} units. "
                        f"Territory {rules['territory']}. Process index "
                        f"{rules['process_min']:.3f} to {rules['process_max']:.3f}. "
                        "Composition fractions must be nonnegative and sum to one within 0.001. "
                        "Safety documentation and provenance are required. "
                        f"Redistribution {'permitted' if rules['redistribution_allowed'] else 'prohibited'}. ")
                text += ("Authoritative release policy for manufacturing authorization."
                         if kind == "release" else
                         "Context note for request review and documentation; nonauthorizing example only.")
                policies.append({"doc_id": doc_id, "family": family, "revision": f"v{v}",
                                 "type": "release" if kind == "release" else "context",
                                 "text": text, "rules": dict(rules), "fictional": True})
    return policies


HEADERS = ("Material request", "Manufacturing submission", "Supplier release review",
           "Fabrication application", "Engineering authorization")
DOC_PHRASES = (("safety", "provenance"), ("safety_record", "origin_record"),
               ("safety_documentation", "traceability_record"))


def serialize_request(x, group):
    """15 complete template groups = 5 headers × 3 wording/field orders."""
    a, b = DOC_PHRASES[group % 3]
    fields = [f"family {x['family']}", f"policy {x['revision']}",
              f"quantity {x['quantity']}", f"territory {x['territory']}",
              f"process {x['process_index']:.6f}",
              "composition " + " ".join(f"{z:.6f}" for z in x["composition"]),
              f"{a} {'present' if x['safety_documented'] else 'absent'}",
              f"{b} {'present' if x['provenance_present'] else 'absent'}",
              f"redistribution {'requested' if x['redistribute'] else 'not_requested'}"]
    # Fixed orders, shared across records in a group. No label/anomaly/gold ID.
    offset = group % 9
    fields = fields[offset:] + fields[:offset]
    if group % 2:
        fields.reverse()
    return HEADERS[group // 3] + ": " + "; ".join(fields) + "."


def independent_audit(x):
    """Second implementation of label checks; does not call inference rules."""
    f = FAMILIES.index(x["family"])
    v = int(x["revision"][1:])
    faults = []
    if x["safety_documented"] is not True:
        faults.append("safety")
    if not 1 <= x["quantity"] <= 600 - 100 * v - 20 * f:
        faults.append("quota")
    if x["territory"] != ("zone_a" if v % 2 else "zone_b"):
        faults.append("territory")
    if not (.12 + .035*f + .035*v <= x["process_index"] <= .90 - .02*f - .025*v):
        faults.append("process")
    if min(x["composition"]) < 0 or abs(sum(x["composition"]) - 1) > .001:
        faults.append("composition")
    if x["provenance_present"] is not True:
        faults.append("provenance")
    if x["redistribute"] and (f + v) % 2 == 0:
        faults.append("redistribution")
    return faults


def generate_assets(seed=SEED):
    rng = np.random.default_rng(seed)
    staged = []
    for f, family in enumerate(FAMILIES):
        # Each category has 13 per family + 2 cyclic extras: 93 total/category.
        anomalies = [None] * 372 + [name for name in VIOLATIONS for _ in range(13)]
        anomalies += [VIOLATIONS[f], VIOLATIONS[(f + 1) % 7]]
        rng.shuffle(anomalies)
        for anomaly in anomalies:
            v = int(rng.integers(1, 3))
            if anomaly == "redistribution":
                v = 2 if f % 2 == 0 else 1
            rules = policy_rules(f, v)
            fractions = np.round(rng.dirichlet([2, 3, 4]), 6)
            fractions[-1] = round(1 - float(fractions[:-1].sum()), 6)
            x = {"family": family, "revision": f"v{v}",
                 "quantity": int(rng.integers(1, rules["quota"] + 1)),
                 "territory": rules["territory"],
                 "process_index": float(rng.uniform(rules["process_min"], rules["process_max"])),
                 "composition": fractions.tolist(), "safety_documented": True,
                 "provenance_present": True,
                 "redistribute": bool(rng.integers(0, 2)) if rules["redistribution_allowed"] else False}
            if anomaly == "safety":
                x["safety_documented"] = False
            elif anomaly == "quota":
                x["quantity"] = rules["quota"] + int(rng.integers(1, 81))
            elif anomaly == "territory":
                x["territory"] = "zone_b" if rules["territory"] == "zone_a" else "zone_a"
            elif anomaly == "process":
                delta = float(rng.uniform(.002, .08))
                x["process_index"] = (rules["process_min"] - delta if rng.integers(0, 2)
                                      else rules["process_max"] + delta)
            elif anomaly == "composition":
                x["composition"][0] += float(rng.uniform(.015, .15))
            elif anomaly == "provenance":
                x["provenance_present"] = False
            elif anomaly == "redistribution":
                x["redistribute"] = True
            faults = independent_audit(x)
            if faults != ([] if anomaly is None else [anomaly]):
                raise ValueError(f"Generation audit failed: {anomaly}, {faults}")
            staged.append((x, anomaly))
    rng.shuffle(staged)
    assets, targets, folds = [], [], []
    for i, (x, anomaly) in enumerate(staged):
        request_id = f"RMSC-{i+1:05d}"
        group = i % 15
        x["text"] = serialize_request(x, group)
        input_hash = hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        x["request_id"] = request_id
        assets.append(x)
        targets.append({"request_id": request_id, "compliant": int(anomaly is None),
                        "violation": anomaly or "none",
                        "gold_policy_id": policy_id(x["family"], x["revision"]),
                        "input_sha256": input_hash})
        folds.append({"request_id": request_id, "template_group": group, "test_fold": group // 3})
    assert len({t["input_sha256"] for t in targets}) == 3255
    assert Counter(x["family"] for x in assets) == {f: 465 for f in FAMILIES}
    assert Counter(t["violation"] for t in targets) == {"none": 2604, **{v: 93 for v in VIOLATIONS}}
    return assets, targets, folds
