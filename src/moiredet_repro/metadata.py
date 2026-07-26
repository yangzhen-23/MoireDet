"""Schema-versioned metadata for an auditable MoireDet inference run."""

from datetime import datetime, timezone
import platform

import torch
import torchvision


UPSTREAM_URL = "https://github.com/cong-yang/MoireDet"
UPSTREAM_COMMIT = "afde899f3c3beee96160610ee450618136a38f7b"


def build_run_metadata(input_path, prepared, config, service, prediction, requested_device):
    """Build the fixed portable schema before rendering fills in display statistics."""
    info = service.checkpoint_info
    perf = service.last_performance
    manifest = info.manifest
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "path": str(input_path.resolve()),
            "width": prepared.width,
            "height": prepared.height,
            "channel_order": "BGR",
        },
        "model": {
            "name": config.model_name,
            "upstream_repository": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint": {
                "path": str(info.path),
                "filename": manifest.filename,
                "size_bytes": info.size_bytes,
                "sha256": info.sha256,
                "source_type": manifest.source_type,
                "source_reference": manifest.source_reference,
                "retrieved_at": manifest.retrieved_at,
                "provenance_evidence": manifest.provenance_evidence,
                "checkpoint_verified": info.checkpoint_verified,
            },
        },
        "runtime": {
            "device_requested": requested_device,
            "device_resolved": str(service.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(service.device)
                if service.device.type == "cuda"
                else None
            ),
        },
        "preprocessing": {
            "input_size": list(config.input_size),
            "channel_order": config.channel_order,
            "mean": list(config.mean),
            "std": list(config.std),
        },
        "prediction": {
            "shape": list(prediction.shape),
            "dtype": str(prediction.dtype),
            "min": None,
            "max": None,
            "dynamic_range": None,
        },
        "performance": {
            "forward_ms": perf.forward_ms,
            "peak_memory_allocated_bytes": perf.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": perf.peak_memory_reserved_bytes,
        },
        "outputs": {
            "prediction": "prediction.npy",
            "moire_map": "moire_map.png",
            "comparison": "comparison.png",
            "run": "run.json",
        },
    }
