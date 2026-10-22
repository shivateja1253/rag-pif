import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd

sha = lambda t: hashlib.sha1(t.encode()).hexdigest()


def write_run(folder, seed, adv_l3):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    sets = {"standard": ([1] * 10 + [0] * 10), "evasion": [1] * 8, "adversarial": [0] * 10, "ood": [0] * 5}
    for name, lab in sets.items():
        n = len(lab)
        l3 = np.where(np.array(lab) == 1, 0.9, 0.05)
        l2 = np.full(n, 0.1)
        if name == "adversarial":
            l3 = np.array(adv_l3, float); l2 = np.array([0.7] * 5 + [0.1] * 5)   # 5 items L2 would block at 0.5 and 0.65
        pd.DataFrame({"text_sha1": [sha(f"{name}{i}") for i in range(n)], "label": lab, "l1_blocked": False,
                      "l2_score": l2, "l3_prob": l3, "first_blocking_layer": 0}).to_csv(folder / f"decisions_{name}.csv", index=False)


def test_rules_and_aggregation(tmp_path):
    dec = tmp_path / "res" / "decisions"
    for seed, adv in [(1, [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), (2, [0.1] * 10)]:
        write_run(dec / f"C3_both_s{seed}__none", seed, adv)
    asg = pd.DataFrame({"text_sha1": [sha(f"adversarial{i}") for i in range(10)], "set": "adversarial", "half": "heldout",
                        "audit_status": ["clean"] * 8 + ["contested"] * 2})
    asg.to_csv(tmp_path / "asg.csv", index=False)
    r = subprocess.run([sys.executable, "evaluation/scripts/summarize_heldout.py", "--decisions-dir", str(dec),
                        "--assignments", str(tmp_path / "asg.csv"), "--l2-threshold", "0.8"], capture_output=True, text=True)
    out = r.stdout
    assert r.returncode == 0, r.stderr
    # t*=0.8: L2 (score 0.7) never blocks; cascade@0.50: items 0-4 blocked by L2 -> adv_all_fpr 0.5 in both seeds
    line = {ln.split("  ")[0].strip(): ln for ln in out.splitlines() if ln.startswith(("cascade", "L2 off", "L3 alone"))}
    assert "0.500+-0.000" in line["cascade @0.50"]
    # seed 1 has one L3 false positive (item 0), seed 2 none -> mean 0.05 with t* ; sd 0.0707
    assert "0.050+-0.071" in line["cascade @t*"]
    assert (tmp_path / "res" / "heldout_summary.csv").exists()
