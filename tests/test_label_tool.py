import importlib.util
from pathlib import Path

from shottracker.evaluation import load_labels

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("label_shots", ROOT / "scripts" / "label_shots.py")
label_shots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label_shots)


def test_saved_labels_keep_the_template_header_and_load_in_the_evaluator(tmp_path):
    f = tmp_path / "video_01.csv"
    f.write_text("# video_01.mp4 - Outdoors\n# instructions\ntime_s,outcome,note\n")
    comments, rows = label_shots.read_labels(f)
    assert comments == ["# video_01.mp4 - Outdoors", "# instructions"] and rows == []

    label_shots.write_labels(f, comments, [(19.8, "missed", ""), (5.25, "made", "")])
    text = f.read_text()
    assert text.startswith("# video_01.mp4 - Outdoors\n# instructions\ntime_s,outcome,note\n")
    assert [(s.t, s.outcome) for s in load_labels(f)] == [(5.25, "made"), (19.8, "missed")]

    _, again = label_shots.read_labels(f)  # reopening a video resumes its labels
    assert again == [(5.25, "made", ""), (19.8, "missed", "")]
