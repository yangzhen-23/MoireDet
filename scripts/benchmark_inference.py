"""Run the fixed trusted-checkpoint RTX acceptance benchmark."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moiredet_repro.config import load_config
from moiredet_repro.inference import MoireDetInference, set_determinism
from moiredet_repro.preprocessing import prepare_image


class BenchmarkOutputError(OSError):
    """The benchmark report could not be safely published."""


def write_fresh_json(output, result):
    """Publish a finite JSON report only if no output entry already exists."""
    try:
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkOutputError("benchmark result is not finite JSON: {}".format(exc)) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            written = handle.write(payload)
            if written != len(payload):
                raise OSError("incomplete benchmark report write")
    except (FileExistsError, IsADirectoryError, PermissionError) as exc:
        if output.exists():
            raise BenchmarkOutputError(
                "--output must be a fresh, nonexistent file"
            ) from exc
        raise BenchmarkOutputError(
            "could not create benchmark output {}: {}".format(output, exc)
        ) from exc
    except OSError as exc:
        raise BenchmarkOutputError(
            "could not publish benchmark output {}: {}".format(output, exc)
        ) from exc


def parser():
    value = argparse.ArgumentParser(description="MoireDet RTX acceptance benchmark")
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--checkpoint-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", choices=("cuda",), default="cuda")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        parser().error("--output must be a fresh, nonexistent file")
    config = load_config()
    set_determinism(2)
    service = MoireDetInference.from_checkpoint(
        args.checkpoint, args.checkpoint_manifest, config, args.device
    )
    prepared = prepare_image(args.input, config, service.device)
    result = asdict(service.benchmark_tensor(prepared.tensor, warmup=5, iterations=20))
    result.update(
        {
            "input": str(args.input.resolve()),
            "checkpoint_sha256": service.checkpoint_info.sha256,
            "checkpoint_verified": service.checkpoint_info.checkpoint_verified,
        }
    )
    try:
        write_fresh_json(args.output, result)
    except BenchmarkOutputError as exc:
        parser().error(str(exc))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
