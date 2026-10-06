import json
from pathlib import Path
from types import SimpleNamespace

import run_sweep as rs


def fake_results(adv=0.2):
    ab = {}
    for split, key, val in [("standard", "detection_rate", 1.0), ("standard", "fpr", 0.0),
                            ("evasion", "detection_rate", 1.0), ("adversarial", "fpr", adv), ("ood", "fpr", 0.0)]:
        for c in rs.COMBOS:
            ab.setdefault(split, {}).setdefault(c, {})[key] = val
    return {"ablation": ab, "cascade": []}


def test_plan_pairs_c0_c3_first_then_single_source():
    p = rs.plan([42, 1])
    assert p[:4] == [("C0_none", 42), ("C3_both", 42), ("C0_none", 1), ("C3_both", 1)]
    assert p[4:] == [("C1_adv", 42), ("C2_ent", 42), ("C1_adv", 1), ("C2_ent", 1)]


def test_compact_extracts_all_layers_and_metrics():
    m = rs.compact(fake_results(0.19))
    assert set(m) == set(rs.COMBOS) and m["L3"]["adv_fpr"] == 0.19 and m["L1+L2+L3"]["std_det"] == 1.0


def test_run_one_skips_finished_and_writes_json(tmp_path, monkeypatch):
    work, res = tmp_path / "work", tmp_path / "res"
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        if "train_l3.py" in cmd[1]:
            out = Path(cmd[cmd.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
            (out / "final").mkdir(exist_ok=True)
            (out / "train_meta.json").write_text(json.dumps({"train_rows": 1173, "best_val": {"eval_f1": 0.99}}))
        else:
            tag = cmd[cmd.index("--tag") + 1]
            d = work / "runs" / f"{tag}_20260101T000000Z"; d.mkdir(parents=True)
            (d / "results.json").write_text(json.dumps(fake_results()))

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    a = SimpleNamespace(results_dir=str(res), work_dir=str(work), data_dir="data/v2_1", extra_file="x.csv",
                        faiss="f.faiss", device="cpu", select="last")
    assert rs.run_one("C3_both", 7, a) == "done"
    assert "--select" in calls[0] and calls[0][calls[0].index("--select") + 1] == "last"
    assert "--extra-sources" in calls[0] and "adversarial,synthetic_enterprise" in calls[0]
    assert (res / "C3_both_s7.json").exists() and not (work / "C3_both_s7").exists()
    n = len(calls)
    assert rs.run_one("C3_both", 7, a) == "skipped" and len(calls) == n
    assert rs.run_one("C0_none", 7, a) == "done" and "--extra" not in calls[n]


def test_summarize_prints_mean_and_std(tmp_path, capsys):
    for s, adv in [(1, 0.1), (2, 0.3)]:
        (tmp_path / f"C0_none_s{s}.json").write_text(json.dumps(
            {"cond": "C0_none", "seed": s, "metrics": rs.compact(fake_results(adv))}))
    rs.summarize(str(tmp_path))
    out = capsys.readouterr().out
    assert "0.200+-0.141" in out and "C0_none" in out
