"""Convert report WebP previews to PDFs; optionally refresh previews from a run."""

import argparse
from pathlib import Path

from matplotlib.figure import Figure
from PIL import Image

from src.analysis_report import REPORT_FIGURES
from src.config import PROJECT_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Analysis output directory to refresh WebP previews")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "docs/figures/analysis",
                        help="Report figure directory (default: docs/figures/analysis)")
    args = parser.parse_args(argv)
    sources = [args.input / (stem + ".png") if args.input else
               args.output / (Path(stem).name + ".webp") for _, stem in REPORT_FIGURES]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("Missing generated figures: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    preview_bytes = pdf_bytes = 0
    for source in sources:
        destination = args.output / (source.stem + ".webp")
        if args.input:
            with Image.open(source) as figure:
                figure.save(destination, format="WEBP", lossless=True, quality=100, method=6, exact=True)
        preview_bytes += destination.stat().st_size
        pdf = destination.with_suffix(".pdf")
        with Image.open(destination) as image:
            # Embed the existing pixels losslessly at the analysis figures' 180 dpi.
            figure = Figure(figsize=(image.width / 180, image.height / 180), dpi=180)
            figure.figimage(image.convert("RGB"), origin="upper")
            figure.savefig(pdf, format="pdf", dpi=180)
            figure.clear()
        pdf_bytes += pdf.stat().st_size
    print(f"Exported {len(REPORT_FIGURES)} figure pairs "
          f"({preview_bytes / 1024:.0f} KiB WebP, {pdf_bytes / 1024:.0f} KiB PDF) to {args.output}")


if __name__ == "__main__":
    main()
