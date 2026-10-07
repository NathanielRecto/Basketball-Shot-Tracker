import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from shottracker.config import ShotConfig
from shottracker.export import golden_case, params, replay_matches, round_frames, sim_frames

ROOT = Path(__file__).resolve().parent.parent


def test_params_cover_every_setting_and_follow_the_code():
    p = params()
    assert set(p["shot"]) == {f.name for f in fields(ShotConfig)}
    assert p["shot"]["arm_margin"] == ShotConfig().arm_margin
    assert p["tracker"]["switch_speed"] == 12.0 and p["tracker"]["exit_wait_s"] == 1.5
    assert p["static_filter"]["window_s"] == 8.0 and p["detector"]["conf"] == 0.15
    json.dumps(p)  # plain JSON


def test_inputs_are_rounded_and_filtered_before_anything_is_computed():
    frames = round_frames([(0.1234567, [(1.004, 2.006, 3.0, 4.0, 0.5), (5, 5, 6, 6, 0.1)])])
    assert frames == [(0.123457, [(1.0, 2.01, 3.0, 4.0, 0.5)])]  # the 0.1-confidence box is below the detector threshold


def test_synthetic_case_covers_the_hard_parts_and_survives_a_json_round_trip():
    hoop, frames = sim_frames()
    case = json.loads(json.dumps(golden_case("sim", hoop, round_frames(frames), "test")))
    outcomes = [e["outcome"] for e in case["expect"]["events"]]
    assert len(outcomes) == 7 and "made" in outcomes and "missed" in outcomes
    assert any(f["expect"]["switched"] for f in case["frames"])  # the tracker leaves the wandering "head"
    assert {"arm", "rattle", "shot", "track_switch"} <= {m.split(":")[0] for _, m in case["expect"]["log"]}
    assert replay_matches(case) is None


def test_a_changed_expectation_is_caught():
    hoop, frames = sim_frames()
    case = json.loads(json.dumps(golden_case("sim", hoop, round_frames(frames)[:300], "test")))
    tracked = next(f for f in case["frames"] if f["expect"]["ball"])
    tracked["expect"]["ball"][0] += 1e-9
    assert replay_matches(case) is not None


def test_export_script_refuses_unlicensed_footage(tmp_path):
    dbg = tmp_path / "debug.json"
    dbg.write_text(json.dumps({"video": "data/eval_videos/video_01.mp4", "hoop": [0, 0, 10, 10], "frames": []}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_app_fixtures.py"), "--out", str(tmp_path / "out"),
                          "--debug-json", str(dbg)], capture_output=True, text=True)
    assert out.returncode != 0 and "no licence" in (out.stdout + out.stderr)


def test_export_script_writes_params_cases_and_manifest(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_app_fixtures.py"), "--out", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["cases"][0]["name"] == "sim_all_shot_types" and manifest["cases"][0]["shots"] == 7
    assert (tmp_path / "params.json").is_file() and (tmp_path / "golden" / "sim_all_shot_types.json").is_file()
