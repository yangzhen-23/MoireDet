class MoireDetReproError(RuntimeError):
    """Base class for expected, user-actionable reproduction failures."""


class ConfigurationError(MoireDetReproError):
    pass


class UpstreamError(MoireDetReproError):
    pass


class InputImageError(MoireDetReproError):
    pass


class CheckpointError(MoireDetReproError):
    pass


class DeviceError(MoireDetReproError):
    pass


class InferenceError(MoireDetReproError):
    pass


class OutputError(MoireDetReproError):
    pass
