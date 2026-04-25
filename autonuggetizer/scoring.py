"""
Nugget-based scoring metrics — Vstrict and Astrict.

Reproduces the scoring formulas from Pradeep et al. (SIGIR 2025), Section 3.4.

  Vstrict = (# vital nuggets with "support") / (# vital nuggets)
  Astrict = (# all nuggets with "support")   / (# all nuggets)

Partial support is NOT counted in strict metrics.
"""


def vstrict(importance: list[str], assignment: list[str]) -> float:
    """Fraction of vital nuggets that receive full 'support'."""
    vital_indices = [i for i, imp in enumerate(importance) if imp == "vital"]
    if not vital_indices:
        return 0.0
    supported = sum(1 for i in vital_indices if assignment[i] == "support")
    return supported / len(vital_indices)


def astrict(importance: list[str], assignment: list[str]) -> float:
    """Fraction of all nuggets that receive full 'support'."""
    if not assignment:
        return 0.0
    supported = sum(1 for a in assignment if a == "support")
    return supported / len(assignment)


def compute_scores(importance: list[str], assignment: list[str]) -> dict:
    """Compute all nugget-based evaluation metrics."""
    n_total = len(assignment)
    n_vital = sum(1 for imp in importance if imp == "vital")
    n_okay = n_total - n_vital

    n_support = sum(1 for a in assignment if a == "support")
    n_partial = sum(1 for a in assignment if a == "partial_support")
    n_not = sum(1 for a in assignment if a == "not_support")

    return {
        "vstrict": vstrict(importance, assignment),
        "astrict": astrict(importance, assignment),
        "nuggets_total": n_total,
        "nuggets_vital": n_vital,
        "nuggets_okay": n_okay,
        "support_count": n_support,
        "partial_support_count": n_partial,
        "not_support_count": n_not,
    }
