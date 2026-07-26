"""Command-line entry point for one verified MoireDet inference run."""

import argparse
import sys
from pathlib import Path
from typing import Dict

from .config import load_config
from .errors import MoireDetReproError
from .inference import MoireDetInference, select_device, set_determinism
from .metadata import build_run_metadata
from .preprocessing import prepare_image
from .rendering import preflight_output_dir, write_output_bundle


def build_parser() -> argparse.ArgumentParser:
    """Build the public one-command CLI parser."""
    parser = argparse.ArgumentParser(description="Official MoireDet single-image reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="run one trusted-checkpoint inference")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--checkpoint-manifest", type=Path, default=None)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer.add_argument("--config", type=Path, default=None)
    return parser


def run_infer(args) -> Dict[str, Path]:
    """Run the trusted end-to-end pipeline and atomically publish its bundle."""
    output = preflight_output_dir(args.output)
    config = load_config(args.config)
    set_determinism(2)
    resolved_device = select_device(args.device)
    print("[1/5] reading and preprocessing {}".format(args.input.resolve()))
    prepared = prepare_image(args.input, config, resolved_device)
    print("[2/5] verifying checkpoint provenance and SHA-256")
    service = MoireDetInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.checkpoint_manifest,
        config=config,
        device=args.device,
    )
    print("[3/5] running official TripleBranchWithSpecificConv")
    prediction = service.predict_tensor(prepared.tensor)
    print("[4/5] building run metadata")
    metadata = build_run_metadata(args.input, prepared, config, service, prediction, args.device)
    print("[5/5] writing output bundle")
    paths = write_output_bundle(
        output, prepared.original_bgr, prediction, metadata, config.constant_epsilon
    )
    for name, path in paths.items():
        print("{}: {}".format(name, path))
    return paths


def main(argv=None) -> int:
    """Execute the CLI and translate expected project errors to exit code 2."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "infer":
            run_infer(args)
            return 0
    except MoireDetReproError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print("UNEXPECTED ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
