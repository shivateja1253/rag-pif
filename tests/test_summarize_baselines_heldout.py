import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd

sha = lambda t: hashlib.sha1(t.encode()).hexdigest()


def write(run, name, probs):
    labels = {"standard": [1] * 4 + [0] * 4, "evasion": [1] * 4, "adversarial": [0] * 4, "ood": [0] * 2}
    for s, lab in labels.items():
        n = len(lab)
        pd.DataFrame({"text_sha1": [sha(f"{s}{i}") for i in range(n)], "label": lab,
                      "p_injection": probs[s]}).to_csv(run / f"{name}_{s}.csv", index=False)


def test_thresholds_clean_subset_and_newest_run(tmp_path):
    runs = tmp_path / "runs"
    old, new = runs / "baselines_none_20260101T000000Z", runs / "baselines_none_20260102T000000Z"
    old.mkdir(parents=True); new.mkdir(parents=True)
    write(old, "det", {"standard": [0.0] * 8, "evasion": [0.0] * 4, "adversarial": [0.0] * 4, "ood": [0.0] * 2})
    write(new, "det", {"standard": [0.95, 0.95, 0.6, 0.1, 0.55, 0.0, 0.0, 0.0], "evasion": [0.95, 0.95, 0.6, 0.1],
                       "adversarial": [0.95, 0.6, 0.1, 0.1], "ood": [0.0, 0.7]})
    pd.DataFrame({"text_sha1": [sha(f"adversarial{i}") for i in range(4)], "set": "adversarial", "half": "heldout",
                  "audit_status": ["clean", "clean", "contested", "clean"]}).to_csv(tmp_path / "asg.csv", index=False)
    r = subprocess.run([sys.executable, "evaluation/scripts/summarize_baselines_heldout.py", "--runs-dir", str(runs),
                        "--assignments", str(tmp_path / "asg.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(runs / "baselines_heldout_summary.csv").set_index("thr")
    # thr 0.5: std_det 3/4, std_fpr 1/4, evas 3/4, adv_all 2/4, adv_clean (items 0,1,3) 2/3, ood 1/2
    assert df.loc[0.5, "std_det"] == 0.75 and df.loc[0.5, "std_fpr"] == 0.25 and df.loc[0.5, "adv_all_fpr"] == 0.5
    assert abs(df.loc[0.5, "adv_clean_fpr"] - 2 / 3) < 1e-9 and df.loc[0.5, "ood_fpr"] == 0.5
    # thr 0.9: only 0.95 scores count
    assert df.loc[0.9, "std_det"] == 0.5 and df.loc[0.9, "adv_all_fpr"] == 0.25 and df.loc[0.9, "ood_fpr"] == 0.0
