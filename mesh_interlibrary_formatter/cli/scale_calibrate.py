from __future__ import annotations

import argparse
from pathlib import Path

from mesh_interlibrary_formatter.adapters.trimesh_adapter import load_with_trimesh, to_trimesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_interlibrary_formatter scale-calibrate",
        description="Load a mesh, apply a uniform scale factor, and write the result.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input mesh path (.stl/.ply/.obj).")
    parser.add_argument("-o", "--output", required=True, help="Output mesh path for scaled geometry.")
    parser.add_argument("--scale", type=float, required=True, help="Uniform scale factor to apply.")
    parser.add_argument("--print-summary", action="store_true", help="Print mesh summaries before/after scaling.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.scale <= 0:
        raise SystemExit("--scale must be > 0")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input mesh not found: {input_path}")

    mesh = load_with_trimesh(str(input_path))
    scaled = mesh.apply_scale(args.scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_trimesh(scaled).export(str(output_path))

    if args.print_summary:
        print(f"Input : {mesh.summary()}")
        print(f"Output: {scaled.summary()}")

    print(f"Wrote scaled mesh to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
