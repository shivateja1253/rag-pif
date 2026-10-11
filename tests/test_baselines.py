import numpy as np
import pytest

import run_baselines as bl


def test_injection_index_from_label_names():
    assert bl.injection_index({0: "SAFE", 1: "INJECTION"}) == 1
    assert bl.injection_index({0: "injection", 1: "benign"}) == 0
    assert bl.injection_index({0: "LABEL_0", 1: "LABEL_1"}) == 1
    assert bl.injection_index({"0": "BENIGN", "1": "MALICIOUS"}) == 1


def test_injection_index_refuses_to_guess():
    with pytest.raises(ValueError):
        bl.injection_index({0: "yes", 1: "no"})


def test_evaluate_probs_threshold_and_single_class_sets():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.95, 0.60, 0.20, 0.55, 0.10, 0.05])
    m5 = bl.evaluate_probs(y, p, 0.5)
    m9 = bl.evaluate_probs(y, p, 0.9)
    assert (m5["tp"], m5["fp"], m5["fn"], m5["tn"]) == (2, 1, 1, 2)
    assert (m9["tp"], m9["fp"], m9["fn"], m9["tn"]) == (1, 0, 2, 3)
    only_benign = bl.evaluate_probs(np.zeros(4, int), np.array([0.6, 0.1, 0.2, 0.7]), 0.5)
    assert only_benign["fpr"] == 0.5 and "detection_rate" not in only_benign


def test_headline_picks_the_right_metric_per_set():
    sets = {"standard": {"detection_rate": 0.9, "fpr": 0.1}, "evasion": {"detection_rate": 0.5},
            "adversarial": {"fpr": 0.3}, "ood": {"fpr": 0.2}}
    assert bl.headline(sets) == (0.9, 0.1, 0.5, 0.3, 0.2)
