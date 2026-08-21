from __future__ import annotations

import argparse
from pathlib import Path

from mesh_interlibrary_formatter.calibration import compute_uniform_scale
from mesh_interlibrary_formatter.adapters.trimesh_adapter import load_with_trimesh, to_trimesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_interlibrary_formatter scale-calibrate",
        description="Load a mesh, apply a uniform scale factor, and write the result.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input mesh path (.stl/.ply/.obj).")
    parser.add_argument("-o", "--output", required=True, help="Output mesh path for scaled geometry.")
    parser.add_argument("--scale", type=float, help="Explicit uniform scale factor to apply.")
    parser.add_argument("--mesh-distance", type=float, help="Distance measured on the mesh between two landmarks.")
    parser.add_argument(
        "--actual-distance",
        type=float,
        help="Physical distance corresponding to --mesh-distance, in matching output units.",
    )
    parser.add_argument("--print-summary", action="store_true", help="Print mesh summaries before/after scaling.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    has_distance_pair = args.mesh_distance is not None or args.actual_distance is not None
    if args.scale is not None and has_distance_pair:
        raise SystemExit("Use either --scale or the --mesh-distance/--actual-distance pair, not both")
    if args.scale is None and not has_distance_pair:
        raise SystemExit("Provide --scale or both --mesh-distance and --actual-distance")
    if has_distance_pair and (args.mesh_distance is None or args.actual_distance is None):
        raise SystemExit("--mesh-distance and --actual-distance must be provided together")

    try:
        scale = (
            compute_uniform_scale(args.mesh_distance, args.actual_distance)
            if has_distance_pair
            else compute_uniform_scale(1.0, args.scale)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input mesh not found: {input_path}")

    mesh = load_with_trimesh(str(input_path))
    scaled = mesh.apply_scale(scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_trimesh(scaled).export(str(output_path))

    if args.print_summary:
        print(f"Input : {mesh.summary()}")
        print(f"Output: {scaled.summary()}")

    print(f"Applied uniform scale factor: {scale:.10g}")
    print(f"Wrote scaled mesh to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
