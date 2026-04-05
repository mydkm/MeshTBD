from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_interlibrary_formatter scale-calibrate",
        description=(
            "Scale calibration CLI scaffold. This command is intentionally minimal "
            "while the consumable package API is under active development."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        help="Input mesh path (.stl/.ply/.obj).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Output mesh path for scaled geometry.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        help="Uniform scale factor to apply.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    print(
        "scale_calibrate CLI scaffold is present, but full implementation is pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
