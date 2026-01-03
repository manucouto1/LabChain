import warnings

warnings.warn(
    "`framework3` has been renamed to `labchain`. " "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from labchain import *  # noqa: E402, F403
