"""Export compact, lossless report previews from a completed analysis run."""

import argparse
from pathlib import Path

from PIL import Image

from src.analysis_report import REPORT_FIGURES
from src.config import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Analysis output directory")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/figures/analysis",
                        help="Preview directory (default: docs/figures/analysis)")
    args = parser.parse_args(argv)
    sources = [args.input / (stem + ".png") for _, stem in REPORT_FIGURES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("Missing generated figures: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for source in sources:
        destination = args.output / (source.stem + ".webp")
        with Image.open(source) as figure:
            figure.save(destination, format="WEBP", lossless=True, quality=100, method=6, exact=True)
        total += destination.stat().st_size
    print(f"Exported {len(sources)} lossless figures ({total / 1024:.0f} KiB) to {args.output}")


if __name__ == "__main__":
    main()
