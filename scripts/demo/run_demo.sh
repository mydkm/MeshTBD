#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="$ROOT/local_data/demo"
OUTPUT_DIR="$ROOT/local_outputs/demo"

mkdir -p "$OUTPUT_DIR"

uv run python "$ROOT/VoronoiFinal.py" -i "$INPUT_DIR/Just forearm.stl" -o "$OUTPUT_DIR/forearm_demo.stl"
uv run python "$ROOT/VoronoiFinal.py" -i "$INPUT_DIR/3D_foot.stl" -o "$OUTPUT_DIR/foot_demo.ply"
uv run python "$ROOT/VoronoiFinal.py" -i "$INPUT_DIR/Sabrina-revised.stl" -o "$OUTPUT_DIR/nonideal_demo.stl"
uv run python "$ROOT/VoronoiFinal.py" -i "$INPUT_DIR/leg.stl" -o "$OUTPUT_DIR/leg_demo.stl"
