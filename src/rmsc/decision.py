"""Auditable approve/reject/escalate decisions on operational fields only."""
import math


def violations(x, policy, skip_material=False):
    r = policy["rules"]
    errors = []
    if not x["safety_documented"]:
        errors.append("safety")
    if not 1 <= x["quantity"] <= r["quota"]:
        errors.append("quota")
    if x["territory"] != r["territory"]:
        errors.append("territory")
    if not skip_material:
        if not r["process_min"] <= x["process_index"] <= r["process_max"]:
            errors.append("process")
        c = x["composition"]
        if (len(c) != 3 or any(not math.isfinite(z) or z < 0 for z in c)
                or abs(sum(c) - 1) > .001):
            errors.append("composition")
    if not x["provenance_present"]:
        errors.append("provenance")
    if x["redistribute"] and not r["redistribution_allowed"]:
        errors.append("redistribution")
    return errors


def decide(x, policy, guard=True, skip_material=False):
    if policy is None:
        return "escalate", "missing_policy"
    if guard and (policy["type"] != "release" or policy["family"] != x["family"]
                  or policy["revision"] != x["revision"]):
        return "escalate", "mismatched_or_nonauthorizing_policy"
    errors = violations(x, policy, skip_material)
    return ("reject", ";".join(errors)) if errors else ("approve", "all_predicates_satisfied")


def unsafe_fallback(x, policy):
    """Explicitly unsafe stress-test comparator, never the default method."""
    if policy is None or policy["type"] != "release" or policy["family"] != x["family"] or policy["revision"] != x["revision"]:
        return "approve", "UNSAFE_missing_or_mismatched_evidence_fallback"
    return decide(x, policy)


def keyword(x):
    """Reconstruction choice: only literal 'safety absent' is screened."""
    return ("reject", "literal_safety_absent") if "safety absent" in x["text"].lower() else ("approve", "keyword_not_found")
