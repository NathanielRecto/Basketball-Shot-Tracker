"""Compare tracker settings across several development videos at once.

    python scripts/replay_all.py outputs/debug/video_*/debug.json --conf 0.3 0.2 0.15 0.1

For each video and threshold: shots found, shots matching a visible ball arc, and shots with
no arc nearby (likely invented). Arcs come from the detector itself, so this is a debugging
aid, not an accuracy measurement; that needs hand labels (scripts/evaluate.py).
"""
import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from replay_debug import find_arcs, replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("debug_json", nargs="+")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.3, 0.2, 0.15, 0.1])
    ap.add_argument("--listed", default=str(ROOT / "data/eval_videos/listed_counts.csv"))
    a = ap.parse_args()

    files = sorted({f for pattern in a.debug_json for f in glob.glob(pattern)})
    listed = {}
    if Path(a.listed).is_file():
        listed = {f"video_{int(r['video']):02d}": r for r in csv.DictReader(open(a.listed))}

    totals = {c: [0, 0, 0, 0] for c in a.conf}  # shots, on-arc, off-arc, made
    arcs_total = 0
    header = f"{'video':<10}{'listed':>8}{'arcs':>6}" + "".join(f"{'conf ' + str(c):>22}" for c in a.conf)
    print(header + "\n" + " " * 24 + "".join(f"{'shots/on-arc/off/made':>22}" for _ in a.conf))
    for path in files:
        debug = json.loads(Path(path).read_text())
        name = Path(path).parent.name
        arcs = find_arcs(debug)
        arcs_total += len(arcs)
        lst = listed.get(name)
        row = f"{name:<10}{(lst['listed_attempts'] + '/' + lst['listed_made']) if lst else '?':>8}{len(arcs):>6}"
        for c in a.conf:
            events, _, _ = replay(json.loads(Path(path).read_text()), c)
            on = sum(any(arc[0][0] - 0.5 <= e.t_cross <= arc[-1][0] + 2.5 for arc in arcs) for e in events)
            made = sum(e.outcome.value == "made" for e in events)
            tot = totals[c]
            tot[0] += len(events); tot[1] += on; tot[2] += len(events) - on; tot[3] += made
            row += f"{f'{len(events)}/{on}/{len(events) - on}/{made}':>22}"
        print(row)
    n_listed = sum(int(listed[Path(p).parent.name]["listed_attempts"]) for p in files if Path(p).parent.name in listed)
    m_listed = sum(int(listed[Path(p).parent.name]["listed_made"]) for p in files if Path(p).parent.name in listed)
    print(f"{'TOTAL':<10}{f'{n_listed}/{m_listed}':>8}{arcs_total:>6}" + "".join(f"{'/'.join(map(str, totals[c])):>22}" for c in a.conf))


if __name__ == "__main__":
    sys.exit(main())
