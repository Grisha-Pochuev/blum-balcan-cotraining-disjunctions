#!/usr/bin/env python3
"""Disjunction learning from agreement pairs.

Feature indices are 0-based. An object is an iterable of active feature
indices; an unlabelled pair is (left, right); a labelled pair is
(left, right, label), where label is 0 or 1.

The finite consistency solver implements Horn forward chaining on excluded
features. The optional PAC wrapper uses independent, identically distributed
pairs only; it never selects which objects receive labels.

Research implementation, 20 October 2026. No novelty or prize claim.
Python 3.10+, standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

Support = Iterable[int]
Pair = tuple[Support, Support]
LabelledPair = tuple[Support, Support, int]


class InconsistentDataError(ValueError):
    """No disjunction can satisfy all supplied constraints."""


def _support(values: Support, dimension: int) -> tuple[int, ...]:
    result = tuple(sorted(set(values)))
    if any(not isinstance(i, int) or i < 0 or i >= dimension for i in result):
        raise ValueError("Feature indices must be integers in [0, dimension).")
    return result


@dataclass(frozen=True)
class Disjunction:
    dimension: int
    relevant: frozenset[int]
    excluded: frozenset[int]
    # (feature, reason kind, index, side). Side 0 is left, 1 is right.
    # A reason of kind 'pair' means that side is already entirely excluded.
    deletion_certificate: tuple[tuple[int, str, int, int], ...]
    propagation_events: int

    def predict(self, active: Support) -> int:
        return int(any(i in self.relevant for i in active))

    def pair_error(self, left: Support, right: Support, label: int) -> bool:
        return self.predict(left) != label or self.predict(right) != label


def learn_consistent(
    dimension: int,
    unlabelled: Iterable[Pair],
    labelled: Iterable[LabelledPair],
) -> Disjunction:
    """Return the greatest feasible disjunction, or report inconsistency.

    Each active feature occurrence is processed O(1) times after input
    normalization. Sorting an object's feature indices is an implementation
    convenience, not a requirement of the mathematical algorithm.
    Empty pair sides are allowed and correctly trigger immediate propagation.
    """
    if not isinstance(dimension, int) or dimension < 0:
        raise ValueError("dimension must be a nonnegative integer")

    sides: list[tuple[int, ...]] = []
    for left, right in unlabelled:
        sides.extend((_support(left, dimension), _support(right, dimension)))

    training: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for left, right, label in labelled:
        if label not in (0, 1):
            raise ValueError("Labels must be 0 or 1.")
        training.append((_support(left, dimension), _support(right, dimension), label))

    incidence: list[list[int]] = [[] for _ in range(dimension)]
    remaining = [len(side) for side in sides]
    for side_index, side in enumerate(sides):
        for feature in side:
            incidence[feature].append(side_index)

    ready = deque(i for i, count in enumerate(remaining) if count == 0)
    excluded = [False] * dimension
    certificate: list[tuple[int, str, int, int]] = []

    def exclude(feature: int, kind: str, index: int, side: int) -> None:
        if excluded[feature]:
            return
        excluded[feature] = True
        certificate.append((feature, kind, index, side))
        for side_index in incidence[feature]:
            remaining[side_index] -= 1
            if remaining[side_index] == 0:
                ready.append(side_index)

    for label_index, (left, right, label) in enumerate(training):
        if label == 0:
            for side_index, side in enumerate((left, right)):
                for feature in side:
                    exclude(feature, "negative", label_index, side_index)

    events = 0
    while ready:
        empty_side = ready.popleft()
        events += 1
        for feature in sides[empty_side ^ 1]:
            exclude(feature, "pair", empty_side // 2, empty_side % 2)

    model = Disjunction(
        dimension=dimension,
        relevant=frozenset(i for i in range(dimension) if not excluded[i]),
        excluded=frozenset(i for i in range(dimension) if excluded[i]),
        deletion_certificate=tuple(certificate),
        propagation_events=events,
    )
    for left, right, label in training:
        if model.pair_error(left, right, label):
            raise InconsistentDataError("No disjunction satisfies these pairs and labels.")
    return model


def lift_literals(bits: Iterable[int]) -> tuple[int, ...]:
    """Lift x in {0,1}^n to active features among x_i and not x_i.

    Positive feature i represents x_i; feature n+i represents not x_i.
    """
    values = tuple(bits)
    if any(value not in (0, 1) for value in values):
        raise ValueError("Expected a Boolean vector.")
    n = len(values)
    return tuple(i if bit else n + i for i, bit in enumerate(values))


def signed_formula(model: Disjunction, original_dimension: int) -> str:
    """Canonical human-readable signed disjunction (tautology is '1')."""
    if model.dimension != 2 * original_dimension:
        raise ValueError("Wrong lifted dimension.")
    if any(i in model.relevant and original_dimension + i in model.relevant
           for i in range(original_dimension)):
        return "1"
    if not model.relevant:
        return "0"
    return " OR ".join(
        f"x{i + 1}" if i < original_dimension
        else f"NOT x{i - original_dimension + 1}"
        for i in sorted(model.relevant)
    )


@dataclass(frozen=True)
class PACResult:
    model: Disjunction
    unlabelled_pairs: int
    labelled_pairs: int
    phases: int
    stopped_by: str
    # phase, training block size, validation block size, validation errors
    history: tuple[tuple[int, int, int, int], ...]


def learn_pac(
    dimension: int,
    draw_unlabelled: Callable[[], Pair],
    draw_labelled: Callable[[], LabelledPair],
    epsilon: float,
    delta: float,
    tau: float | None = None,
) -> PACResult:
    """Count-free PAC wrapper with the conservative constants in the note.

    The sampling callbacks must supply mutually independent iid pairs from
    the SAME joint distribution. Labels must come from a fixed disjunction
    perfectly compatible with that distribution. The code cannot verify
    these population assumptions.

    With probability at least 1-delta, both prediction error and compatibility
    error meet their respective tolerances, and the label count is at most
    400*(ln(K)+ln(16/delta))/epsilon, with K as defined in the note. The expected
    label count has the same order. Runtime has a deterministic polynomial cap.
    """
    if tau is None:
        tau = epsilon
    if any(not (0.0 < p < 1.0) for p in (epsilon, tau)) or not (0.0 < delta <= 0.5):
        raise ValueError("epsilon and tau must lie in (0, 1), and delta in (0, 1/2]")
    if not isinstance(dimension, int) or dimension < 0:
        raise ValueError("dimension must be a nonnegative integer")
    unlabelled_count = math.ceil((dimension * math.log(2) + math.log(4 / delta)) / tau)
    # Materialize supports once: an oracle is allowed to return generators,
    # and U must remain reusable during every training phase.
    unlabelled = []
    for _ in range(unlabelled_count):
        left, right = draw_unlabelled()
        unlabelled.append((_support(left, dimension), _support(right, dimension)))
    b = math.log(16 / delta)
    cap = 4 * (dimension * math.log(2) + b) / epsilon
    labels_used = 0
    history: list[tuple[int, int, int, int]] = []
    phase = 0
    while True:
        m = math.ceil(64 * (2 ** phase) * b / epsilon)
        training = [draw_labelled() for _ in range(m)]
        labels_used += m
        model = learn_consistent(dimension, unlabelled, training)
        if m >= cap:
            history.append((phase, m, 0, 0))
            return PACResult(model, unlabelled_count, labels_used, phase + 1,
                             "deterministic cap", tuple(history))
        # Validate the stronger event 'at least one view is misclassified'.
        errors = 0
        for _ in range(m):
            left, right, label = draw_labelled()
            errors += model.pair_error(left, right, label)
        labels_used += m
        history.append((phase, m, m, errors))
        if errors / m <= epsilon / 2:
            return PACResult(model, unlabelled_count, labels_used, phase + 1,
                             "independent validation", tuple(history))
        phase += 1


def demo() -> dict:
    # Written with 0-based indices: the target is x1 in 1-based notation.
    pairs = [((1,), (2, 3)), ((3,), (4,)), ((0, 2), (0, 4))]
    labelled = [((1,), (1,), 0), ((0, 2), (0, 4), 1)]
    model = learn_consistent(5, pairs, labelled)
    return {
        "example": "A negative x2 propagates through agreement pairs.",
        "feature_numbering": "Results below are 1-based for readability.",
        "remaining_features": [i + 1 for i in sorted(model.relevant)],
        "excluded_features": [i + 1 for i in sorted(model.excluded)],
        "all_unlabelled_pairs_agree": all(model.predict(a) == model.predict(b)
                                         for a, b in pairs),
        "all_labelled_pairs_correct": all(not model.pair_error(a, b, y)
                                          for a, b, y in labelled),
        "propagation_events": model.propagation_events,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the small propagation example")
    parser.parse_args()
    print(json.dumps(demo(), indent=2))
