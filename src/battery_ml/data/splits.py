"""The original train / primary test / secondary test split from Severson et al. (2019).

After cleaning, batches 1 and 2 hold 84 cells. Ordered by batch and channel index, cells
at even positions (plus the last one) form the primary test set (43) and cells at odd
positions form the training set (41). All 40 batch-3 cells form the secondary test set.
"""

import re

SPLITS = ("train", "primary_test", "secondary_test")


def _cell_order(cell_id):
    batch, index = re.fullmatch(r"b(\d+)c(\d+)", cell_id).groups()
    return int(batch), int(index)


def severson_splits(cell_ids):
    """Return {split name: [cell ids]} for the given cleaned cell ids."""
    ordered = sorted(cell_ids, key=_cell_order)
    b12 = [c for c in ordered if not c.startswith("b3")]
    b3 = [c for c in ordered if c.startswith("b3")]

    test_positions = set(range(0, len(b12), 2)) | {len(b12) - 1}
    return {
        "train": [c for i, c in enumerate(b12) if i not in test_positions],
        "primary_test": [c for i, c in enumerate(b12) if i in test_positions],
        "secondary_test": b3,
    }
