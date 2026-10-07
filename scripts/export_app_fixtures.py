"""Export the settings and golden test cases the phone app's port is checked against.

    python scripts/export_app_fixtures.py
    python scripts/export_app_fixtures.py --debug-json outputs/debug960_det2/session_01/debug.json --name own_session_01

Writes <out>/params.json, <out>/golden/<name>.json and <out>/manifest.json (Python commit, case
sizes and SHA-256 hashes). Copy the folder into the app repo; regenerate whenever the tracker,
shot logic or their settings change. Always included: a synthetic case covering every shot type.
Optional: real cases from debug.json files (scripts/debug_video.py), which hold detections only,
no images. Only export footage you have the rights to publish: the public evaluation videos have
no licence and are refused.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shottracker.export import golden_case, params, replay_matches, round_frames, sim_frames  # noqa: E402

UNLICENSED = ("eval_videos",)  # source paths whose derived data must not be published


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("+uncommitted-src-changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def case_from_debug(path: Path, name: str, start: float, end: float):
    dbg = json.loads(path.read_text())
    src = str(dbg.get("video", path))
    if any(u in src.replace("\\", "/") for u in UNLICENSED):
        raise SystemExit(f"refusing to export {src}: that footage has no licence, so its derived data stays local")
    frames = []
    for f in dbg["frames"]:
        if start <= f["t"] <= end:
            balls = [(x - w / 2, y - h / 2, x + w / 2, y + h / 2, c) for x, y, w, h, c in
                     (b if len(b) == 5 else (b[0], b[1], b[2], b[2], b[3]) for b in f["balls"])]
            frames.append((f["t"] - start, balls))
    settings = {k: dbg.get(k) for k in ("imgsz", "stride", "weights") if dbg.get(k) is not None}
    note = f"{Path(src).name}, {start:g}-{min(end, dbg['frames'][-1]['t']):.1f} s" + (f", detector settings {settings}" if settings else "")
    return golden_case(name, dbg["hoop"], round_frames(frames), note)


def write(out: Path, rel: str, obj) -> dict:
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    p.write_text(text)
    return {"file": rel, "bytes": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "exports" / "app"))
    ap.add_argument("--debug-json", action="append", default=[], help="a debug.json to turn into a real-footage case (repeatable)")
    ap.add_argument("--name", action="append", default=[], help="case name for each --debug-json, in order")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into each real clip to start")
    ap.add_argument("--end", type=float, default=float("inf"), help="seconds into each real clip to stop")
    a = ap.parse_args()
    if len(a.name) not in (0, len(a.debug_json)):
        ap.error("give one --name per --debug-json, or none")

    out = Path(a.out)
    cases = []
    hoop, frames = sim_frames()
    cases.append(golden_case("sim_all_shot_types", hoop, round_frames(frames), "synthetic: scripts' simulator, seed 7, 30 fps"))
    for i, dj in enumerate(a.debug_json):
        name = a.name[i] if a.name else Path(dj).parent.name
        cases.append(case_from_debug(Path(dj), name, a.start, a.end))

    manifest = {"python_commit": git_commit(), "created": date.today().isoformat(),
                "params": write(out, "params.json", params()), "cases": []}
    for c in cases:
        problem = replay_matches(json.loads(json.dumps(c)))
        if problem:
            raise SystemExit(f"{c['name']} does not replay identically after a JSON round trip: {problem}")
        info = write(out, f"golden/{c['name']}.json", c)
        info.update(name=c["name"], source=c["source"], frames=len(c["frames"]), shots=len(c["expect"]["events"]))
        manifest["cases"].append(info)
        print(f"{c['name']}: {info['frames']} frames, {info['shots']} shots called, {info['bytes'] / 1e6:.2f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} (python commit {manifest['python_commit']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
