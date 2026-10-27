#!/usr/bin/env python3
"""Independent exhaustive checks of the actual queue-based implementation.

Run beside cotrain_disjunctions.py.  Tests finite constraint solving, literal
lifting, and replayable deletion certificates.  Does not numerically establish
the distribution-free PAC theorem, novelty, or prize eligibility.
"""

from itertools import combinations_with_replacement
import json
from pathlib import Path
from random import Random

from cotrain_disjunctions import (
    InconsistentDataError, learn_consistent, lift_literals, signed_formula,
)


def active(mask, dimension):
    return tuple(j for j in range(dimension) if mask & (1 << j))


def mask(indices):
    result = 0
    for j in indices:
        result |= 1 << j
    return result


def agrees(s, a, b):
    return bool(s & a) == bool(s & b)


def replay_certificate(model, unlabelled, labelled):
    deleted = set()
    for feature, kind, index, side in model.deletion_certificate:
        assert feature not in deleted
        assert side in (0, 1)
        if kind == "negative":
            a, b, y = labelled[index]
            assert y == 0
            assert feature in (a, b)[side]
        elif kind == "pair":
            pair = unlabelled[index]
            assert set(pair[side]).issubset(deleted)
            assert feature in pair[1 - side]
        else:
            raise AssertionError(("unknown certificate reason", kind))
        deleted.add(feature)
    assert deleted == set(model.excluded)
    assert model.relevant.isdisjoint(model.excluded)
    assert model.relevant | model.excluded == frozenset(range(model.dimension))
    assert model.propagation_events <= 2 * len(unlabelled)


def verify_case(dimension, pairs, labels):
    feasible = [
        s for s in range(1 << dimension)
        if all(agrees(s, a, b) for a, b in pairs)
        and all(bool(s & a) == bool(y) and bool(s & b) == bool(y)
                for a, b, y in labels)
    ]
    unlabelled = [(active(a, dimension), active(b, dimension)) for a, b in pairs]
    labelled = [(active(a, dimension), active(b, dimension), y) for a, b, y in labels]
    try:
        model = learn_consistent(dimension, unlabelled, labelled)
    except InconsistentDataError:
        assert not feasible, (dimension, pairs, labels, feasible)
        return None, 0
    assert feasible, (dimension, pairs, labels, model)
    union = 0
    for s in feasible:
        union |= s
    assert mask(model.relevant) == union
    assert union in feasible
    assert all((s & ~union) == 0 for s in feasible)
    replay_certificate(model, unlabelled, labelled)
    assert all(model.predict(a) == model.predict(b) for a, b in unlabelled)
    assert all(not model.pair_error(a, b, y) for a, b, y in labelled)
    return model, len(feasible)


def eval_formula(formula, bits):
    if formula in ("0", "1"):
        return int(formula)
    values = []
    for term in formula.split(" OR "):
        if term.startswith("NOT x"):
            values.append(not bits[int(term[5:]) - 1])
        else:
            assert term.startswith("x")
            values.append(bool(bits[int(term[1:]) - 1]))
    return int(any(values))


def main():
    report = {"status": "passed", "seed": 7251, "checks": []}
    constraint_cases = 0
    certificate_steps = 0
    feasible_witnesses = 0
    for n in range(1, 4):
        possible_pairs = [(a, b) for a in range(1 << n) for b in range(a, 1 << n)]
        count = 0
        for pairs in combinations_with_replacement(possible_pairs, 2):
            for forbidden in range(1 << n):
                model, witnesses = verify_case(n, pairs, [(forbidden, forbidden, 0)])
                certificate_steps += len(model.deletion_certificate)
                feasible_witnesses += witnesses
                count += 1
        constraint_cases += count
        report["checks"].append({"kind": "exhaustive_two_pairs",
                                  "dimension": n, "cases": count})

    rng = Random(7251)
    for n in range(4, 9):
        for _ in range(2000):
            pairs = [(rng.randrange(1 << n), rng.randrange(1 << n))
                     for _ in range(rng.randrange(1, 30))]
            forbidden = rng.randrange(1 << n)
            model, witnesses = verify_case(n, pairs, [(forbidden, forbidden, 0)])
            certificate_steps += len(model.deletion_certificate)
            feasible_witnesses += witnesses
        constraint_cases += 2000
        report["checks"].append({"kind": "random_constraint_systems",
                                  "dimension": n, "cases": 2000})

    observed_labels = 0
    for n in range(1, 9):
        for _ in range(500):
            target = rng.randrange(1 << n)
            pairs = []
            labels = []
            for j in range(32):
                while True:
                    a = rng.randrange(1 << n)
                    b = rng.randrange(1 << n)
                    if agrees(target, a, b):
                        break
                if j < 20:
                    pairs.append((a, b))
                else:
                    labels.append((a, b, int(bool(target & a))))
            model, witnesses = verify_case(n, pairs, labels)
            assert (target & ~mask(model.relevant)) == 0
            for x in range(1 << n):
                assert not (target & x) or model.predict(active(x, n))
            certificate_steps += len(model.deletion_certificate)
            feasible_witnesses += witnesses
            observed_labels += len(labels)
    constraint_cases += 4000
    report["checks"].append({"kind": "target_generated_labelled_samples",
                              "dimensions": [1, 8], "cases": 4000,
                              "labelled_pairs_checked": observed_labels})

    literal_vectors = 0
    for n in range(0, 7):
        for x in range(1 << n):
            bits = tuple(int(bool(x & (1 << j))) for j in range(n))
            lifted = set(lift_literals(bits))
            assert len(lifted) == n
            for j, value in enumerate(bits):
                assert (j in lifted) == bool(value)
                assert (n + j in lifted) == (not value)
            literal_vectors += 1

    formula_truth_values = 0
    for n in range(1, 5):
        inputs = [tuple(int(bool(x & (1 << j))) for j in range(n))
                  for x in range(1 << n)]
        lifts = [mask(lift_literals(x)) for x in inputs]
        for _ in range(500):
            target = rng.randrange(1 << (2 * n))
            pairs = []
            labels = []
            for j in range(32):
                while True:
                    a, b = rng.choice(lifts), rng.choice(lifts)
                    if agrees(target, a, b):
                        break
                if j < 20:
                    pairs.append((a, b))
                else:
                    labels.append((a, b, int(bool(target & a))))
            model, witnesses = verify_case(2 * n, pairs, labels)
            assert (target & ~mask(model.relevant)) == 0
            formula = signed_formula(model, n)
            for bits in inputs:
                assert eval_formula(formula, bits) == model.predict(lift_literals(bits))
                formula_truth_values += 1
            certificate_steps += len(model.deletion_certificate)
            feasible_witnesses += witnesses
    constraint_cases += 2000
    report["checks"].append({"kind": "signed_literal_samples",
                              "original_dimensions": [1, 4], "cases": 2000,
                              "lift_vectors_checked": literal_vectors,
                              "formula_truth_values_checked": formula_truth_values})

    rejected = 0
    for _ in range(1000):
        n = rng.randrange(0, 7)
        pairs = [(rng.randrange(1 << n), rng.randrange(1 << n))
                 for _ in range(rng.randrange(0, 10))]
        labels = [(rng.randrange(1 << n), rng.randrange(1 << n), rng.randrange(2))
                  for _ in range(rng.randrange(1, 7))]
        model, witnesses = verify_case(n, pairs, labels)
        feasible_witnesses += witnesses
        if model is None:
            rejected += 1
        else:
            certificate_steps += len(model.deletion_certificate)
    constraint_cases += 1000
    report["checks"].append({"kind": "arbitrary_possibly_inconsistent_samples",
                              "dimensions": [0, 6], "cases": 1000,
                              "correctly_rejected": rejected})
    report["total_constraint_systems_checked"] = constraint_cases
    report["feasible_support_witnesses_checked"] = feasible_witnesses
    report["certificate_deletions_replayed"] = certificate_steps
    report["scope"] = ["greatest feasible support", "positive and negative labels",
                       "agreement pairs", "global target domination",
                       "literal lifting", "signed formula truth tables",
                       "inconsistency detection", "certificate replay"]
    path = Path(__file__).with_name("solver_verification_results.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
