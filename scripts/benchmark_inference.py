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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
