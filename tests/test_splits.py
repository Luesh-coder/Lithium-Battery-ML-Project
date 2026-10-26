from battery_ml.data import severson_splits
from battery_ml.data.loader import BATCH1_NO_EOL, BATCH3_NOISY, CONTINUED_CELLS


def cleaned_cell_ids():
    b1 = [f"b1c{i}" for i in range(46) if f"b1c{i}" not in BATCH1_NO_EOL]
    b2 = [f"b2c{i}" for i in range(48) if f"b2c{i}" not in CONTINUED_CELLS]
    b3 = [f"b3c{i}" for i in range(46) if f"b3c{i}" not in BATCH3_NOISY]
    return b3 + b2 + b1  # deliberately out of order


def test_split_sizes_match_paper():
    splits = severson_splits(cleaned_cell_ids())
    assert {name: len(ids) for name, ids in splits.items()} == {
        "train": 41,
        "primary_test": 43,
        "secondary_test": 40,
    }


def test_splits_are_disjoint_and_interleaved():
    splits = severson_splits(cleaned_cell_ids())
    all_ids = [c for ids in splits.values() for c in ids]
    assert len(all_ids) == len(set(all_ids)) == 124
    assert splits["primary_test"][:2] == ["b1c0", "b1c2"]
    assert splits["train"][:2] == ["b1c1", "b1c3"]
    assert splits["primary_test"][-1] == "b2c47"
