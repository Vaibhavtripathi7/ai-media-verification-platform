"""Turn eval/results/ into the numbers in docs/evaluation.md.

    python -m eval.report

Calibration and verdict thresholds are fitted on the validation split and only then
applied to test and reddit, so the reported numbers are not tuned on the data they
describe.
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .score import RESULTS

SERVED = "communityforensics-vit"
BASELINE = "sdxl-detector"
# Each confident verdict band has to be right this often on validation.
TARGET_PRECISION = 0.9
SETS = [
    ("test", "OpenFake test"),
    ("test-recompressed", "OpenFake test, recompressed"),
    ("reddit", "Reddit, in the wild"),
]


@dataclass(frozen=True)
class Calibration:
    slope: float
    intercept: float

    def __call__(self, logit: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.slope * logit + self.intercept)))


def load(candidate: str, name: str) -> dict[str, dict]:
    """Scored rows by id. Images the service refused are reported separately."""
    path = RESULTS / candidate / f"{name}.jsonl"
    if not path.exists():
        return {}
    return {row["id"]: row for row in map(json.loads, path.open()) if "logit" in row}


def rejected(candidate: str, name: str) -> list[dict]:
    path = RESULTS / candidate / f"{name}.jsonl"
    if not path.exists():
        return []
    return [row for row in map(json.loads, path.open()) if "rejected" in row]


def arrays(rows: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    is_ai = np.array([r["is_ai"] for r in rows.values()])
    logit = np.array([r["logit"] for r in rows.values()])
    return is_ai, logit


def fit_calibration(is_ai: np.ndarray, logit: np.ndarray) -> Calibration:
    """Platt scaling: a one-feature logistic regression on the raw logit."""
    model = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), is_ai)
    return Calibration(float(model.coef_[0, 0]), float(model.intercept_[0]))


def pick_thresholds(is_ai: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Lowest AI cut-off and highest real cut-off that each meet TARGET_PRECISION."""
    candidates = np.unique(p)
    ai_cut = next((t for t in candidates if is_ai[p >= t].mean() >= TARGET_PRECISION), 1.0)
    real_cut = next(
        (t for t in candidates[::-1] if (~is_ai[p <= t]).mean() >= TARGET_PRECISION), 0.0
    )
    return float(ai_cut), float(min(real_cut, ai_cut))


def expected_calibration_error(is_ai: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    which = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(
            abs(is_ai[which == b].mean() - p[which == b].mean()) * (which == b).mean()
            for b in range(bins)
            if (which == b).any()
        )
    )


def share(mask: np.ndarray, within: np.ndarray) -> float:
    return float(mask[within].mean()) if within.any() else float("nan")


def pct(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x * 100:.1f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    is_ai, logit = arrays(load(SERVED, "validation"))
    calibrate = fit_calibration(is_ai, logit)
    ai_cut, real_cut = pick_thresholds(is_ai, calibrate(logit))
    print(f"Fitted on {len(is_ai)} validation images:\n")
    print(f"CALIBRATION_SLOPE={calibrate.slope:.4f}")
    print(f"CALIBRATION_INTERCEPT={calibrate.intercept:.4f}")
    print(f"AI_THRESHOLD={ai_cut:.3f}")
    print(f"REAL_THRESHOLD={real_cut:.3f}")

    ranking, bands = [], []
    for name, label in SETS:
        rows = load(SERVED, name)
        if not rows:
            continue
        is_ai, logit = arrays(rows)
        p = calibrate(logit)
        says_ai, says_real = p >= ai_cut, p <= real_cut
        raw = 1 / (1 + np.exp(-logit))
        ranking.append(
            [
                label,
                str(len(rows)),
                f"{roc_auc_score(is_ai, logit):.3f}",
                f"{expected_calibration_error(is_ai, raw):.3f}",
                f"{expected_calibration_error(is_ai, p):.3f}",
            ]
        )
        bands.append(
            [
                label,
                pct((says_ai | says_real).mean()),
                pct(share(is_ai, says_ai)),
                pct(share(~is_ai, says_real)),
                pct(share(says_ai, is_ai)),
                pct(share(says_real, ~is_ai)),
            ]
        )

    print("\n## Ranking and calibration\n")
    print(table(["Set", "Images", "AUROC", "ECE raw", "ECE calibrated"], ranking))
    print("\n## Verdict bands\n")
    header = ["Set", "Answered", "Likely AI right", "No sign right", "AI caught", "Real cleared"]
    print(table(header, bands))

    print("\n## Per source, OpenFake test\n")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in load(SERVED, "test").values():
        by_source[row["source"]].append(row)
    rows = []
    for source, items in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        p = calibrate(np.array([r["logit"] for r in items]))
        label = items[0]["is_ai"]
        right = (p >= ai_cut).mean() if label else (p <= real_cut).mean()
        wrong = (p <= real_cut).mean() if label else (p >= ai_cut).mean()
        rows.append([source, "AI" if label else "real", str(len(items)), pct(right), pct(wrong)])
    print(table(["Source", "Label", "Images", "Right verdict", "Wrong verdict"], rows))

    print("\n## Against the previous detector, each at its own 0.5 cut-off\n")
    rows = []
    for name, label in SETS:
        line = [label]
        for candidate in (BASELINE, SERVED):
            data = load(candidate, name)
            if not data:
                line += ["n/a", "n/a"]
                continue
            is_ai, logit = arrays(data)
            line += [f"{roc_auc_score(is_ai, logit):.3f}", pct(((logit >= 0) == is_ai).mean())]
        rows.append(line)
    print(table(["Set", f"{BASELINE} AUROC", "accuracy", f"{SERVED} AUROC", "accuracy"], rows))

    print("\n## Reduced-size JPEG decoding\n")
    full, draft = load(f"{SERVED}-full-decode", "test"), load(SERVED, "test")
    ids = sorted(full.keys() & draft.keys())
    if ids:
        is_ai = np.array([full[i]["is_ai"] for i in ids])
        full_logit = np.array([full[i]["logit"] for i in ids])
        draft_logit = np.array([draft[i]["logit"] for i in ids])
        diff = np.abs(calibrate(full_logit) - calibrate(draft_logit))
        header = ["Images", "AUROC full decode", "AUROC draft", "Mean abs Δp", "Max abs Δp"]
        line = [
            str(len(ids)),
            f"{roc_auc_score(is_ai, full_logit):.4f}",
            f"{roc_auc_score(is_ai, draft_logit):.4f}",
            f"{diff.mean():.4f}",
            f"{diff.max():.4f}",
        ]
        print(table(header, [line]))

    print("\n## Refused by the service's own limits\n")
    for name in ["validation", *(name for name, _ in SETS)]:
        for row in rejected(SERVED, name):
            print(f"- {name} {row['id']} ({row['source']}): {row['rejected']}")

    print("\n## Scoring time per image during the eval run\n")
    rows = []
    for candidate in (BASELINE, SERVED):
        ms = [r["ms"] for r in load(candidate, "test").values()]
        if ms:
            rows.append([candidate, f"{np.median(ms):.0f} ms", f"{np.percentile(ms, 95):.0f} ms"])
    print(table(["Candidate", "Median", "p95"], rows))


if __name__ == "__main__":
    main()
