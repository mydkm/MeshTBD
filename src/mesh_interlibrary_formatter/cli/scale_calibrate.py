from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_interlibrary_formatter scale-calibrate",
        description=(
            "Compatibility scaffold retained while MeshTBD migrates from the old "
            "formatter package into the new meshtbd package."
        ),
    )
    parser.add_argument("-i", "--input", required=False, help="Input mesh path.")
    parser.add_argument("-o", "--output", required=False, help="Output mesh path.")
    parser.add_argument("--scale", type=float, required=False, help="Uniform scale factor to apply.")
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    print(
        "mesh_interlibrary_formatter is now a compatibility layer. "
        "Use VoronoiFinal.py or meshtbd.cli.voronoi_cast for the primary workflow."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
