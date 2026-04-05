from __future__ import annotations

import argparse
from pathlib import Path

from ..calibration import default_scaled_polydata_path
from ..models import CalibrationInput, ExportConfig, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="VoronoiFinal.py",
        description="Generate a cast-like mesh with Voronoi openings from an input surface mesh.",
    )
    parser.add_argument(
        "stl",
        nargs="?",
        help="Path to input mesh file (.stl/.ply).",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        default=None,
        help="Path to input mesh file (.stl/.ply). Alias for positional stl.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default=None,
        help="Path to final output mesh (.stl/.ply).",
    )
    parser.add_argument(
        "--right-click",
        action="store_true",
        help="Use right-click instead of left-click.",
    )
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Automatically close the picker window after the second pick.",
    )
    parser.add_argument(
        "--picker",
        default="cell",
        choices=["hardware", "cell", "point", "volume"],
        help="VTK picker type used for clicks.",
    )
    parser.add_argument(
        "--scaled-polydata-out",
        default=None,
        help="Optional output path for the scaled intermediate mesh.",
    )
    parser.add_argument(
        "--real-mm",
        type=float,
        default=None,
        help="Known real-world distance between the selected landmarks in millimeters.",
    )
    parser.add_argument(
        "--landmark-vertices",
        nargs=2,
        type=int,
        metavar=("V0", "V1"),
        help="Skip interactive picking and use existing vertex ids to measure the geodesic.",
    )
    parser.add_argument(
        "--no-color-preview",
        action="store_true",
        help="Skip the mid-pipeline PyVista color preview window.",
    )
    return parser


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_output.stl")


def build_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> PipelineConfig:
    if args.stl and args.input and Path(args.stl) != Path(args.input):
        parser.error("Input path conflict: positional 'stl' and '--input' differ. Use only one.")

    input_arg = args.input or args.stl
    if input_arg is None:
        parser.error("Missing input mesh. Provide positional 'stl' or '-i/--input'.")

    input_path = Path(input_arg).expanduser()
    output_path = Path(args.output).expanduser() if args.output else default_output_path(input_path)
    scaled_polydata_out = (
        Path(args.scaled_polydata_out).expanduser()
        if args.scaled_polydata_out
        else default_scaled_polydata_path(input_path)
    )

    if args.real_mm is not None and args.real_mm <= 0:
        parser.error("--real-mm must be positive.")

    landmark_vertices = tuple(args.landmark_vertices) if args.landmark_vertices else None

    calibration = CalibrationInput(
        input_path=input_path,
        left_click=not args.right_click,
        auto_close=args.auto_close,
        picker=args.picker,
        landmark_vertices=landmark_vertices,
        real_world_distance_mm=args.real_mm,
    )
    export = ExportConfig(
        output_path=output_path,
        scaled_polydata_out=scaled_polydata_out,
        preview_color_mesh=not args.no_color_preview,
    )
    return PipelineConfig(
        input_path=input_path,
        calibration=calibration,
        export=export,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args, parser)

    from ..pipeline.cast_pipeline import run_pipeline

    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
