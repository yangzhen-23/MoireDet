from moiredet_repro import __version__
from moiredet_repro.errors import CheckpointError, InputImageError, OutputError, UpstreamError


def test_package_exports_version_and_domain_errors():
    assert __version__ == "0.1.0"
    for error_type in (CheckpointError, InputImageError, OutputError, UpstreamError):
        assert issubclass(error_type, RuntimeError)
