#!/usr/bin/env python3
"""Independent small-instance audit of disjunction agreement propagation.

The verifier uses a straightforward set-deletion implementation and compares its
answer with exhaustive enumeration of every support.  This is an independent
empirical check of the finite constraint solver, not a substitute for the PAC
proof and not a check of novelty or of the prize's present status.
"""

from itertools import combinations_with_replacement
import json
from pathlib import Path
from random import Random


def bits(mask, n):
    return {j for j in range(n) if mask & (1 << j)}


def agree(support, a, b):
    return bool(support & a) == bool(support & b)


def set_deletion_closure(n, pairs, forbidden):
    """Deliberately simple independent solver, without queues or counters."""
    support = set(range(n)) - bits(forbidden, n)
    pair_sets = [(bits(a, n), bits(b, n)) for a, b in pairs]
    while True:
        old = support.copy()
        for a, b in pair_sets:
            if not support.intersection(a):
                support.difference_update(b)
            if not support.intersection(b):
                support.difference_update(a)
        if support == old:
            return sum(1 << j for j in support)


def audit_system(n, pairs, forbidden):
    feasible = [
        s for s in range(1 << n)
        if not (s & forbidden) and all(agree(s, a, b) for a, b in pairs)
    ]
    union = 0
    for s in feasible:
        union |= s
    got = set_deletion_closure(n, pairs, forbidden)
    assert got == union, (n, pairs, forbidden, got, feasible)
    assert got in feasible, (n, pairs, forbidden, got)
    assert all((s & ~got) == 0 for s in feasible)
    # The preceding assertion verifies retention of every possible compatible
    # target, and hence preservation of every positive target-labelled pair.
    return got, len(feasible)


def main():
    report = {"seed": 7251, "checks": [], "status": "passed"}
    total_cases = 0
    total_target_witnesses = 0
    for n in range(1, 4):
        # Orientation and repeated pair choices are immaterial to agreement.
        pairs = [(a, b) for a in range(1 << n) for b in range(a, 1 << n)]
        count = 0
        witnesses = 0
        for system in combinations_with_replacement(pairs, 2):
            for forbidden in range(1 << n):
                _, number_feasible = audit_system(n, system, forbidden)
                count += 1
                witnesses += number_feasible
        report["checks"].append({
            "kind": "exhaustive_two_pairs", "dimension": n,
            "systems_with_forbidden_masks": count,
            "feasible_target_witnesses": witnesses,
        })
        total_cases += count
        total_target_witnesses += witnesses

    rng = Random(7251)
    for n in range(4, 9):
        witnesses = 0
        for _ in range(2000):
            system = [
                (rng.randrange(1 << n), rng.randrange(1 << n))
                for _ in range(rng.randrange(1, 30))
            ]
            forbidden = rng.randrange(1 << n)
            _, number_feasible = audit_system(n, system, forbidden)
            witnesses += number_feasible
        report["checks"].append({
            "kind": "random_systems", "dimension": n,
            "systems_with_forbidden_masks": 2000,
            "feasible_target_witnesses": witnesses,
        })
        total_cases += 2000
        total_target_witnesses += witnesses

    labelled_cases = 0
    observed_labels = 0
    for n in range(1, 9):
        for _ in range(500):
            target = rng.randrange(1 << n)
            system = []
            labels = []
            forbidden = 0
            for j in range(32):
                while True:
                    a = rng.randrange(1 << n)
                    b = rng.randrange(1 << n)
                    if agree(target, a, b):
                        break
                if j < 20:
                    system.append((a, b))
                else:
                    y = bool(target & a)
                    labels.append((a, b, y))
                    if not y:
                        forbidden |= a | b
            got, _ = audit_system(n, system, forbidden)
            assert (target & ~got) == 0
            for a, b, y in labels:
                assert bool(got & a) == y
                assert bool(got & b) == y
                observed_labels += 1
            # Check the global one-sided property explicitly on every point.
            for x in range(1 << n):
                assert not (target & x) or bool(got & x)
            labelled_cases += 1
    report["checks"].append({
        "kind": "target_generated_labelled_samples",
        "dimension_range": [1, 8],
        "systems": labelled_cases,
        "labelled_pairs_checked": observed_labels,
        "checks": ["greatest_feasible_support", "target_retention",
                   "both_views_match_all_labels", "global_no_false_negatives"],
    })
    report["unlabelled_constraint_systems_checked"] = total_cases
    report["feasible_target_witnesses_checked"] = total_target_witnesses
    report["labelled_constraint_systems_checked"] = labelled_cases
    report["total_constraint_systems_checked"] = total_cases + labelled_cases
    destination = Path(__file__).with_name("independent_verify_results.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
