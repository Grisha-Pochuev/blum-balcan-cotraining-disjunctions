#!/usr/bin/env python3
"""Execution checks for the capped, independent-validation wrapper.
These are not substitutes for the probability bounds in the manuscript.
"""
import json
import math
from pathlib import Path
from random import Random
from cotrain_disjunctions import learn_pac


def run_case(dimension, epsilon, delta, seed):
    rng = Random(seed)
    calls = {"unlabelled": 0, "labelled": 0}

    def draw_unlabelled():
        i = rng.randrange(dimension)
        calls["unlabelled"] += 1
        # One-shot supports verify that cached U is materialized.
        return iter((i,)), iter((i,))

    def draw_labelled():
        i = rng.randrange(dimension)
        calls["labelled"] += 1
        return iter((i,)), iter((i,)), 0

    result = learn_pac(dimension, draw_unlabelled, draw_labelled,
                       epsilon=epsilon, delta=delta)
    # Distribution is uniform on (e_i,e_i), target is constant zero.
    true_error = len(result.model.relevant) / dimension
    assert result.unlabelled_pairs == calls["unlabelled"]
    assert result.labelled_pairs == calls["labelled"]
    assert result.labelled_pairs == sum(m + v for _, m, v, _ in result.history)
    assert result.phases == len(result.history)
    assert true_error <= epsilon
    assert result.labelled_pairs <= 400 * (
        dimension * math.log(2) + math.log(16 / delta)) / epsilon
    return {
        "dimension": dimension,
        "epsilon": epsilon,
        "delta": delta,
        "seed": seed,
        "true_error": true_error,
        "unlabelled_pairs": result.unlabelled_pairs,
        "labelled_pairs": result.labelled_pairs,
        "phases": result.phases,
        "stopped_by": result.stopped_by,
        "history": result.history,
    }


def main():
    cap = run_case(4, 0.2, 0.1, 318)
    validation = run_case(4096, 0.49, 0.5, 20261020)
    assert cap["stopped_by"] == "deterministic cap"
    assert validation["stopped_by"] == "independent validation"
    assert validation["phases"] > 1
    report = {
        "status": "passed",
        "checks": [cap, validation],
        "scope": ["cap", "rejection and continuation", "validation acceptance",
                  "sample accounting", "one-shot unlabelled supports"],
        "limitation": "Seeded execution checks do not establish a distribution-free theorem.",
    }
    Path(__file__).with_name("wrapper_smoke_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

