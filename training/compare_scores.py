"""Compare a served model with its training weights on the same validation states.

  python3 training/compare_scores.py runs/ft_q1/score_hf.json runs/score_served.json

The first file is score_hf.py's output, the second the --out file of
score_ollama.py; rows are joined on seed and tick. Reports how often both pick
the same option, the largest and median probability difference, and the
training-side margin between the top two options where the picks differ, so
near-ties can be told apart from real disagreements. Standard library only.
"""
import argparse
import json
import statistics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hf")
    ap.add_argument("served")
    args = ap.parse_args()
    with open(args.hf) as f:
        hf = {(r["seed"], r["tick"]): r["probabilities"] for r in json.load(f)}
    with open(args.served) as f:
        served = json.load(f)["details"]
    same, diffs, margins = 0, [], []
    for d in served:
        hp = hf.get((d["seed"], d["tick"]))
        if hp is None:
            continue
        diffs.append(max(abs(hp[k] - d["probabilities"][k]) for k in hp))
        if max(hp, key=hp.get) == d["choice"]:
            same += 1
        else:
            top = sorted(hp.values(), reverse=True)
            margins.append(round(top[0] - top[1], 4))
    print(json.dumps({
        "compared": len(diffs),
        "same_choice": same,
        "max_prob_diff": round(max(diffs), 4),
        "median_prob_diff": round(statistics.median(diffs), 4),
        "margins_where_choice_differs": sorted(margins),
    }))


if __name__ == "__main__":
    main()
