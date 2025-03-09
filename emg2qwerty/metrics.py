# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from typing import Any

import Levenshtein
import torch
from torchmetrics import Metric

from emg2qwerty.data import LabelData


KEY_MAPPINGS = {
    ' ': 10,
    '`': 1,
    '\t': 1,
    '1': 2,
    'q': 1,
    'a': 1,
    'z': 1,
    '2': 2,
    'w': 2,
    's': 2,
    'x': 2,
    '3': 3,
    'e': 3,
    'd': 3,
    '4': 4,
    'r': 4,
    'f': 4,
    'v': 4,
    '5': 4,
    't': 4,
    'g': 4,
    'b': 4,
    '6': 5,
    'y': 5,
    'h': 5,
    'n': 5,
    '7': 5,
    'u': 5,
    'j': 5,
    'm': 5,
    '8': 6,
    'i': 6,
    'k': 6,
    ',': 6,
    '9': 6,
    'o': 7,
    'l': 7,
    '.': 7,
    '0': 7,
    'p': 7,
    '[': 7,
    ';': 8,
    '/': 8,
    '\\': 8,
    "'": 8,
    ']': 8,
}


class CharacterErrorRates(Metric):
    """Character-level error rates metrics based on Levenshtein edit-distance
    between the predicted and target sequences.

    Returns a dictionary with the following metrics:
    - ``CER``: Character Error Rate
    - ``IER``: Insertion Error Rate
    - ``DER``: Deletion Error Rate
    - ``SER``: Substitution Error Rate
    - ``FER``: Finger Error Rate
    - ``AER``: "Almost" Finger Error Rate

    As an instance of ``torchmetric.Metric``, synchronization across all GPUs
    involved in a distributed setting is automatically performed on every call
    to ``compute()``."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)

        self.add_state("insertions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("deletions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("substitutions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_len", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("finger_substitutions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("almost_finger_substitutions", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: LabelData, target: LabelData) -> None:
        # Use Levenshtein.editops rather than Levenshtein.distance to
        # break down errors into insertions, deletions and substitutions.
        editops = Levenshtein.editops(prediction.text, target.text)
        edits = Counter(op for op, _, _ in editops)

        # Update running counts
        self.insertions += edits["insert"]
        self.deletions += edits["delete"]
        self.substitutions += edits["replace"]

        for op in editops:
            if op[0] == "replace":
                p_char = prediction.text[op[1]]
                t_char = target.text[op[2]]
                if KEY_MAPPINGS.get(p_char, 15) != KEY_MAPPINGS.get(t_char, 16):
                    self.finger_substitutions += 1
                if abs(KEY_MAPPINGS.get(p_char, 15) - KEY_MAPPINGS.get(t_char, 20)) > 1:
                    self.almost_finger_substitutions += 1

        self.target_len += len(target)

    def compute(self) -> dict[str, float]:
        def _error_rate(errors: torch.Tensor) -> float:
            return float(errors.item() / self.target_len.item() * 100.0)

        return {
            "CER": _error_rate(self.insertions + self.deletions + self.substitutions),
            "IER": _error_rate(self.insertions),
            "DER": _error_rate(self.deletions),
            "SER": _error_rate(self.substitutions),
            "FER": _error_rate(self.finger_substitutions),
            "AER": _error_rate(self.almost_finger_substitutions),
        }
