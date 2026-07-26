from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type
import sys

from .config import InferenceConfig
from .errors import UpstreamError


@dataclass(frozen=True)
class UpstreamPaths:
    repo_root: Path
    performer_root: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_upstream(root: Optional[Path] = None) -> UpstreamPaths:
    base = Path(root) if root is not None else project_root()
    repo = base / "MoireDet"
    performer = repo / "script"
    missing = [path for path in (repo / "lib", performer / "performer_pytorch") if not path.exists()]
    if missing:
        raise UpstreamError(
            "Missing pinned upstream component(s): {}".format(
                ", ".join(map(str, missing))
            )
        )
    return UpstreamPaths(repo.resolve(), performer.resolve())


def activate_upstream_imports(paths: UpstreamPaths) -> None:
    for path in reversed((paths.repo_root, paths.performer_root)):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def build_official_model(
    config: InferenceConfig, model_class: Optional[Type] = None
) -> "torch.nn.Module":
    if model_class is None:
        paths = resolve_upstream()
        activate_upstream_imports(paths)
        from lib.models.model import TripleBranchWithSpecificConv

        model_class = TripleBranchWithSpecificConv
    return model_class(dict(config.model_args))
