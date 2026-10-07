import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from shottracker.evaluation import (
    LabeledShot, PredictedShot, evaluate_video, load_labels, match_shots, normalize_outcome,
    parse_time, summarize, wilson_interval,
)

ROOT = Path(__file__).resolve().parent.parent


def L(t, o):
    return LabeledShot(t, o)


def P(t, o):
    return PredictedShot(t, o)


# ---- parsing ----------------------------------------------------------------------------

@pytest.mark.parametrize("text,secs", [("12.5", 12.5), ("1:23", 83), ("1:23.4", 83.4), ("0:01:23", 83), (" 7 ", 7)])
def test_parse_time(text, secs):
    assert parse_time(text) == pytest.approx(secs)


def test_parse_time_rejects_garbage():
    with pytest.raises(ValueError):
        parse_time("1:2:3:4")
    with pytest.raises(ValueError):
        parse_time("abc")


@pytest.mark.parametrize("word,expected", [("Made", "made"), ("x", "missed"), ("1", "made"), (" MISS ", "missed")])
def test_normalize_outcome(word, expected):
    assert normalize_outcome(word) == expected


def test_normalize_outcome_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_outcome("maybe")


def test_load_labels_ignores_comments_and_blank_template_rows(tmp_path):
    f = tmp_path / "video_01.csv"
    f.write_text("# header comment\n\ntime_s,outcome,note\n19.8,missed,hit rim\n1:05,made,\n,,\n")
    labels = load_labels(f)
    assert [(s.t, s.outcome, s.note) for s in labels] == [(19.8, "missed", "hit rim"), (65.0, "made", "")]


def test_load_labels_empty_template_is_empty(tmp_path):
    f = tmp_path / "video_02.csv"
    f.write_text("# c\ntime_s,outcome,note\n")
    assert load_labels(f) == []


def test_load_labels_reads_shot_type(tmp_path):
    f = tmp_path / "session_02.csv"
    f.write_text("time_s,outcome,shot_type,note\n5,made,3,\n9,missed,FT,\n12,made,,\n")
    assert [s.shot_type for s in load_labels(f)] == ["3PT", "FT", ""]
    f.write_text("time_s,outcome,shot_type\n5,made,layup\n")
    with pytest.raises(ValueError):
        load_labels(f)


def test_load_labels_names_the_bad_row(tmp_path):
    f = tmp_path / "video_03.csv"
    f.write_text("time_s,outcome,note\n5,made,\n6,perhaps,\n")
    with pytest.raises(ValueError, match=r"video_03.csv.*row 2"):
        load_labels(f)


# ---- matching ---------------------------------------------------------------------------

def test_matching_is_one_to_one_and_closest_first():
    labels = [L(10.0, "made")]
    preds = [P(10.9, "made"), P(10.2, "made")]
    pairs, miss, extra = match_shots(labels, preds, tol_s=1.5)
    assert pairs == [(0, 1)] and miss == [] and extra == [0]


def test_matching_respects_tolerance():
    pairs, miss, extra = match_shots([L(10.0, "made")], [P(12.0, "made")], tol_s=1.5)
    assert pairs == [] and miss == [0] and extra == [0]


# ---- scoring ----------------------------------------------------------------------------

def test_perfect_video():
    r = evaluate_video("v", [L(5, "made"), L(15, "missed")], [P(5.3, "made"), P(14.6, "missed")])
    assert (r.matched, r.correct, r.false_pos, r.false_neg, r.errors) == (2, 2, 0, 0, [])
    assert r.make_count_error == 0


def test_every_error_kind_is_reported():
    labels = [L(5, "made"), L(15, "made"), L(25, "missed")]
    preds = [P(5, "made"), P(15, "missed"), P(40, "made")]  # right, wrong outcome, extra; the 25s shot is not found
    r = evaluate_video("v", labels, preds)
    assert (r.matched, r.correct, r.false_pos, r.false_neg) == (2, 1, 1, 1)
    assert sorted(e["kind"] for e in r.errors) == ["extra_shot", "missed_shot", "wrong_outcome"]
    assert r.made_labeled == 2 and r.made_predicted == 2 and r.make_count_error == 0  # offsetting errors hide in counts


def test_results_by_shot_type():
    labels = [LabeledShot(5, "made", shot_type="3PT"), LabeledShot(15, "made", shot_type="3PT"),
              LabeledShot(25, "missed", shot_type="FT"), LabeledShot(35, "missed")]
    r = evaluate_video("v", labels, [P(5, "made"), P(15, "missed"), P(25, "missed"), P(35, "missed")])
    assert r.by_shot_type == {"3PT": [1, 2], "FT": [1, 1]}  # untagged shots are left out


def test_summary_numbers():
    a = evaluate_video("a", [L(5, "made"), L(15, "made")], [P(5, "made"), P(15, "missed")])
    b = evaluate_video("b", [L(5, "missed"), L(15, "missed"), L(25, "made")], [P(5, "made"), P(15, "missed"), P(60, "made")])
    s = summarize([a, b])
    assert (s.labeled, s.predicted, s.matched) == (5, 5, 4)
    assert s.precision == pytest.approx(4 / 5) and s.recall == pytest.approx(4 / 5)
    assert s.outcome_accuracy == pytest.approx(2 / 4)       # of the 4 matched, 2 correct
    assert s.end_to_end_accuracy == pytest.approx(2 / 5)    # the unfound shot counts as wrong
    assert (s.made_to_missed, s.missed_to_made) == (1, 1)


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.404, 0.596), abs=0.002)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0)[0] != wilson_interval(0, 0)[0]  # nan


# ---- the script, end to end -------------------------------------------------------------

def test_evaluate_script_end_to_end(tmp_path):
    labels, pred = tmp_path / "labels", tmp_path / "pred"
    labels.mkdir()
    (labels / "video_01.csv").write_text("time_s,outcome\n5,made\n15,missed\n")
    (labels / "video_02.csv").write_text("time_s,outcome\n")  # not labelled yet
    (labels / "video_03.csv").write_text("time_s,outcome\n8,made\n")  # labelled but never run
    (pred / "video_01").mkdir(parents=True)
    shots = [{"t_cross": 5.2, "outcome": "made", "reason": "through_hoop"},
             {"t_cross": 15.1, "outcome": "made", "reason": "off_target"}]
    (pred / "video_01" / "summary.json").write_text(json.dumps({"shots": shots}))

    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--labels", str(labels), "--pred", str(pred),
         "--meta", str(tmp_path / "none.csv"), "--author", str(tmp_path / "none.csv")],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "skipped video_02: no labels yet" in out.stdout
    assert "skipped video_03: no predictions" in out.stdout
    assert "END-TO-END accuracy: 50.0%" in out.stdout

    with open(pred / "errors.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["kind"], r["label"], r["predicted"]) for r in rows] == [("wrong_outcome", "missed", "made")]
    assert json.loads((pred / "evaluation.json").read_text())["summary"]["labeled"] == 2


def test_evaluate_script_with_nothing_to_do(tmp_path):
    (tmp_path / "labels").mkdir()
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--labels", str(tmp_path / "labels"),
                          "--pred", str(tmp_path / "pred"), "--meta", "x", "--author", "x"], capture_output=True, text=True)
    assert out.returncode == 1 and "Nothing to evaluate" in out.stdout


def test_compare_runs_script_scores_runs_on_the_same_shots(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "session_02.csv").write_text("time_s,outcome,shot_type,note\n5,made,3PT,\n15,missed,FT,\n")
    (labels / "session_03.csv").write_text("time_s,outcome\n8,made\n")  # only in one run -> left out
    for run, outcomes in (("old", ["missed", "missed"]), ("new", ["made", "missed"])):
        (tmp_path / run / "session_02").mkdir(parents=True)
        shots = [{"t_cross": t, "outcome": o} for t, o in zip((5.1, 15.2), outcomes)]
        (tmp_path / run / "session_02" / "summary.json").write_text(json.dumps({"shots": shots}))
    (tmp_path / "new" / "session_03").mkdir()
    (tmp_path / "new" / "session_03" / "summary.json").write_text(json.dumps({"shots": []}))

    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--labels", str(labels),
         "--run", f"old={tmp_path / 'old'}", "--run", f"new={tmp_path / 'new'}"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "left out session_03" in out.stdout
    lines = {ln.split()[0]: ln for ln in out.stdout.splitlines() if ln.startswith(("old ", "new "))}
    assert "50.0%" in lines["old"] and "100.0%" in lines["new"]


def test_compare_runs_warns_when_runs_used_different_settings(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "video_01.csv").write_text("time_s,outcome\n5,made\n")
    for run, imgsz in (("a", 960), ("b", 1280)):
        d = tmp_path / run / "video_01"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"shots": [{"t_cross": 5.0, "outcome": "made"}]}))
        (d / "settings.json").write_text(json.dumps({"imgsz": imgsz, "stride": 1}))
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--labels", str(labels),
         "--run", f"a={tmp_path / 'a'}", "--run", f"b={tmp_path / 'b'}"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "WARNING" in out.stdout and "imgsz on video_01" in out.stdout
