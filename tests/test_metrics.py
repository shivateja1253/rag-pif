import numpy as np
import pytest

import run_benchmark as rb


def test_wilson_matches_known_values():
    lo, hi = rb.wilson(150, 150)
    assert 0.975 < lo < 0.977 and hi == pytest.approx(1.0)
    lo, hi = rb.wilson(0, 131)
    assert lo == 0.0 and 0.02 < hi < 0.03


def test_metrics_counts_and_f1():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    m = rb.metrics(y, p)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert m["fpr"] == pytest.approx(1 / 5)


def test_single_class_sets_have_no_f1():
    m = rb.metrics([1, 1, 1], [1, 1, 0])
    assert "f1" not in m and "fpr" not in m and m["detection_rate"] == pytest.approx(2 / 3)
    m = rb.metrics([0, 0, 0], [1, 0, 0])
    assert "f1" not in m and "detection_rate" not in m and m["fpr"] == pytest.approx(1 / 3)


def test_cascade_equals_or_of_layers_and_first_layer_attribution():
    sc = {"b1": np.array([True, False, False, False]),
          "s2": np.array([0.9, 0.8, 0.1, 0.1]),
          "p3": np.array([0.9, 0.1, 0.9, 0.1])}
    combos, first = rb.derive(sc)
    assert first.tolist() == [1, 2, 3, 0]
    assert combos["L1+L2+L3"].tolist() == [True, True, True, False]
    assert combos["L2+L3"].tolist() == [True, True, True, False]
    assert combos["L1"].tolist() == [True, False, False, False]
