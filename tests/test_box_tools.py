import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sampler = _load("sample_box_frames")
labeler = _load("box_labeler")


def _frame(i, balls, tracked=None, fps=60.0):
    return {"t": round(i / fps, 3), "balls": balls, "tracked": tracked}


def test_frames_are_sorted_into_what_the_detector_finds_hard():
    ball = [500, 400, 30, 30, 0.8]
    frames = [
        _frame(0, [ball], tracked=[500, 400]),            # ordinary, followed
        _frame(120, []),                                  # in the shot window (0.5-3.1 s), nothing seen
        _frame(150, [ball], tracked=[500, 400]),          # in the shot window, seen
        _frame(300, [[500, 10, 30, 30, 0.6]], [500, 10]), # touching the top edge
        _frame(600, [ball, [900, 700, 30, 30, 0.7]], [500, 400]),  # two "balls", away from shots
        _frame(660, [[900, 700, 30, 30, 0.7]]),           # a "ball" the tracker did not follow
    ]
    cats = sampler.categorize(frames, shot_times=[2.5], fps=60.0)
    assert cats["top"] == [300]
    assert cats["missed"] == [120]
    assert cats["flight"] == [120, 150]
    assert cats["extra"] == [600, 660]
    assert cats["random"] == [0, 120, 150, 300, 600, 660]


def test_picked_frames_keep_their_distance_and_respect_the_total():
    cats = {"top": list(range(0, 100)), "missed": [], "flight": list(range(0, 1000)), "extra": [], "random": list(range(0, 5000))}
    picked = sampler.pick(cats, n=50, min_gap=8, seed=1)
    idx = [i for i, _ in picked]
    assert len(picked) == 50
    assert all(b - a >= 8 for a, b in zip(idx, idx[1:]))
    assert sum(r == "top" for _, r in picked) == 5  # 10 % quota
    assert sampler.pick(cats, n=50, min_gap=8, seed=1) == picked  # reproducible


def test_yolo_boxes_survive_a_save_and_load_and_are_clipped_to_the_image(tmp_path):
    w, h = 1920, 1080
    lines = sampler.yolo_lines([(0, 100, 50, 140, 90), (1, 1900, -10, 1950, 60), (0, 5, 5, 6, 6)], w, h)
    assert len(lines) == 2  # the 1 px box is dropped
    p = tmp_path / "f.txt"
    p.write_text("\n".join(lines) + "\n")
    boxes = labeler.read_yolo(p, w, h)
    assert [round(v, 1) for v in boxes[0]] == [0, 100.0, 50.0, 140.0, 90.0]
    assert [round(v, 1) for v in boxes[1]] == [1, 1900.0, 0.0, 1920.0, 60.0]
    labeler.write_yolo(p, boxes, w, h)
    assert p.read_text().splitlines() == lines


def test_clicking_inside_the_hoop_box_selects_the_ball_in_it():
    hoop = [1, 400, 200, 560, 420]
    ball = [0, 460, 230, 500, 270]
    assert labeler.box_at([hoop, ball], 480, 250) == 1
    assert labeler.box_at([hoop, ball], 420, 400) == 0
    assert labeler.box_at([hoop, ball], 10, 10) is None


def test_test_sessions_are_refused_by_split(tmp_path):
    p = tmp_path / "sessions.csv"
    p.write_text("# comment\nsession,original_file,split,notes\n1,a.MOV,dev,\n6,b.MOV,test,\n")
    assert sampler.session_splits(p) == {1: "dev", 6: "test"}
