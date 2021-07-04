"""
scmdata, simple data handling for simple climate model data
"""
from ._version import get_versions  # isort:skip

__version__ = get_versions()["version"]
del get_versions


from scmdata.run import ScmRun, run_append  # noqa: F401, E402
from scmdata.ensemble import ScmEnsemble, ensemble_append  # noqa: F401, E402

# TODO: remove in next major release
REQUIRED_COLS = ScmRun.required_cols  # noqa
