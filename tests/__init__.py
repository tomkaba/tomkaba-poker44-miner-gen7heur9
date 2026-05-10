import sys
import types


def _ensure_mock_bittensor() -> None:
    if "bittensor" in sys.modules:
        return

    class _MockLogging:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

        def trace(self, *args, **kwargs):
            pass

        def success(self, *args, **kwargs):
            pass

        def set_config(self, *args, **kwargs):
            pass

        def set_trace(self, *args, **kwargs):
            pass

    class _Unavailable:
        def __init__(self, *args, **kwargs):  # pragma: no cover - protective stub
            raise RuntimeError(
                "bittensor runtime components are not available in the test harness"
            )

        def __call__(self, *args, **kwargs):  # pragma: no cover - protective stub
            raise RuntimeError(
                "bittensor runtime components are not available in the test harness"
            )

    mock_bt = types.SimpleNamespace(
        logging=_MockLogging(),
        Wallet=_Unavailable,
        Subtensor=_Unavailable,
        Axon=_Unavailable,
        Synapse=_Unavailable,
        __version__="0.0-test",
    )
    sys.modules["bittensor"] = mock_bt


_ensure_mock_bittensor()
