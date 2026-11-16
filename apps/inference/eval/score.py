"""Score one candidate on one split and append predictions to eval/results/.

    python -m eval.score communityforensics-vit test
    python -m eval.score communityforensics-vit test --recompress

Reruns skip ids that are already in the output file, so a long run can be resumed.
Images that cannot be decoded, or that the service would refuse, are recorded with
the reason instead of a score.
"""

import argparse
import json
import time
from pathlib import Path

from app.imaging import ImageTooLarge, InvalidImage

from .candidates import CANDIDATES
from .openfake import SPLITS, recompress, samples

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=CANDIDATES)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--recompress", action="store_true")
    parser.add_argument("--limit", type=int, help="score only the first N images")
    args = parser.parse_args()

    name = f"{args.split}-recompressed" if args.recompress else args.split
    out = RESULTS / args.candidate / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()

    score = CANDIDATES[args.candidate]()
    with out.open("a") as f:
        for i, sample in enumerate(samples(args.split)):
            if i == args.limit:
                break
            if sample.id in done:
                continue
            row = {"id": sample.id, "is_ai": sample.is_ai, "source": sample.source}
            try:
                data = recompress(sample.data) if args.recompress else sample.data
                start = time.perf_counter()
                row["logit"] = round(score(data), 4)
                row["ms"] = round((time.perf_counter() - start) * 1000, 1)
            # OSError is how Pillow itself reports an unreadable file, which the
            # recompression step and the transformers baseline run into directly.
            except (InvalidImage, ImageTooLarge, OSError) as exc:
                row["rejected"] = str(exc)
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"{args.candidate} {name}: {i}", flush=True)


if __name__ == "__main__":
    main()
