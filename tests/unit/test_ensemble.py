import datetime as dt
import re

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

from scmdata.ensemble import ScmEnsemble, ensemble_append
from scmdata.run import ScmRun

inplace_param = pytest.mark.parametrize("inplace", [True, False])


@pytest.fixture(scope="function")
def ensemble(scm_run):
    return ScmEnsemble(
        [
            scm_run,
            scm_run.filter(variable="Primary Energy").interpolate(
                [dt.datetime(y, 1, 1) for y in range(2010, 2026, 5)]
            ),
        ]
    )


def test_ensemble(scm_run):
    ensemble = ScmEnsemble([scm_run, scm_run])

    assert len(ensemble) == len(scm_run) * 2
    assert ensemble.num_timeseries == len(scm_run) * 2
    assert ensemble.num_runs == 2


def test_ensemble_num_timeseries(ensemble):
    assert ScmEnsemble().num_timeseries == 0

    assert ensemble.num_timeseries == sum([len(r) for r in ensemble.runs])
    assert ensemble.num_timeseries > 0


def test_ensemble_repr(ensemble):
    res = repr(ensemble)

    assert "scmdata.ensemble.ScmEnsemble" in res
    assert "runs: {}".format(ensemble.num_runs) in res
    assert "timeseries: {}".format(ensemble.num_timeseries) in res


def test_ensemble_copy(ensemble):
    ensemble_copy = ensemble.copy(deep=False)
    assert id(ensemble) != id(ensemble_copy)

    for l, r in zip(ensemble.runs, ensemble_copy.runs):
        assert id(l) == id(r)

    ensemble_copy = ensemble.copy(deep=True)
    assert id(ensemble) != id(ensemble_copy)

    for l, r in zip(ensemble.runs, ensemble_copy.runs):
        assert id(l) != id(r)


@pytest.mark.parametrize("item", ("time", "year", "scenario", "variable"))
def test_ensemble_getitem(ensemble, item):
    if item == "year":
        assert isinstance(ensemble[item], pd.Series)
        npt.assert_array_almost_equal(ensemble["year"], [2005, 2010, 2015, 2020, 2025])
    elif item == "time":
        assert isinstance(ensemble[item], pd.Series)
        assert np.all(
            ensemble["time"]
            == [dt.datetime(y, 1, 1) for y in [2005, 2010, 2015, 2020, 2025]]
        )
    else:
        assert isinstance(ensemble[item], pd.Series)
        exp = pd.concat([r[item] for r in ensemble.runs])
        pdt.assert_series_equal(ensemble[item], exp)


def test_ensemble_timeseries(ensemble):
    res = ensemble.timeseries()

    npt.assert_array_almost_equal(ensemble.runs[0]["year"], [2005, 2010, 2015])
    npt.assert_array_almost_equal(ensemble.runs[1]["year"], [2010, 2015, 2020, 2025])
    npt.assert_array_almost_equal(res.columns.year, [2005, 2010, 2015, 2020, 2025])

    # index should be in order
    assert res.index.names == sorted(res.index.names)

    def _get_timeseries(idx, r):
        ts = r.timeseries()
        ts["run_id"] = idx
        ts = ts.set_index("run_id", append=True)

        return ts

    exp = pd.concat([_get_timeseries(run_id, r) for run_id, r in ensemble])
    exp.index = exp.index.reorder_levels(sorted(exp.index.names))

    pd.testing.assert_frame_equal(res, exp)

    assert np.all(np.isnan(exp.xs(0, level="run_id").loc[:, "2020-01-01":"2025-01-01"]))

    # Check that subset is the same
    pdt.assert_frame_equal(
        ensemble.runs[0].timeseries().reset_index(),
        res.xs(0, level="run_id").dropna(how="all", axis="columns").reset_index(),
        check_like=True,  # To remove once #97 is resolved
    )


def test_ensemble_timeseries_empty():
    res = ScmEnsemble().timeseries()
    assert res.empty


def test_ensemble_timeseries_overlapping(ensemble):
    ensemble.runs[0]["run_id"] = 72
    with pytest.warns(UserWarning,):
        res = ensemble.timeseries()

    run_ids = res.index.get_level_values(level="run_id")
    pdt.assert_index_equal(run_ids, pd.Index([0, 0, 0, 1, 1], name="run_id"))

    ensemble.meta_col = "ensemble_member"
    with pytest.warns(None) as record:
        res = ensemble.timeseries()
    assert len(record) == 0

    run_ids = res.index.get_level_values(level="run_id")
    pdt.assert_index_equal(
        run_ids,
        pd.Index([72, 72, 72, pd.NA, pd.NA], name="run_id"),
        exact=False,
        check_exact=False,
    )
    run_ids = res.index.get_level_values(level="ensemble_member")
    pdt.assert_index_equal(run_ids, pd.Index([0, 0, 0, 1, 1], name="ensemble_member"))


@inplace_param
def test_ensemble_filter_basic(ensemble, inplace):
    res = ensemble.filter(scenario="a_scenario2", inplace=inplace)

    if inplace:
        assert res is None
        res = ensemble
    else:
        assert id(res) != id(ensemble)
        assert len(res) != id(ensemble)

    assert np.all(res["scenario"] == "a_scenario2")


@inplace_param
def test_ensemble_append(ensemble, scm_run, inplace):
    orig = ensemble.copy()
    res = ensemble_append([ensemble, ensemble.copy()], inplace=inplace)

    if inplace:
        assert res is None
        res = ensemble
    else:
        assert id(res) != id(ensemble)

    assert isinstance(res, ScmEnsemble)

    assert len(res) == len(orig) * 2


def test_ensemble_append_wrong_first(ensemble, scm_run):
    with pytest.raises(TypeError, match="Can only append inplace to an ScmEnsemble"):
        ensemble_append([scm_run, ensemble], inplace=True)


def test_ensemble_append_mixed(ensemble, scm_run):
    res = ensemble_append([ensemble, ensemble])
    assert isinstance(res, ScmEnsemble)
    assert res.num_runs == 4

    res = ensemble_append([ensemble, scm_run])
    assert isinstance(res, ScmEnsemble)
    assert res.num_runs == 3

    res = ensemble.append(scm_run)
    assert isinstance(res, ScmEnsemble)
    assert res.num_runs == 3

    res = ensemble_append([scm_run])
    assert isinstance(res, ScmEnsemble)
    assert res.num_runs == 1


def test_ensemble_append_empty():
    with pytest.raises(ValueError, match="Nothing to append"):
        ensemble_append([])


def test_ensemble_append_nonlist():
    with pytest.raises(TypeError, match="ensemble_or_runs is not a list"):
        ensemble_append(1)


@pytest.mark.parametrize(
    "obj", ["a", 1, np.asarray([1, 2, 3]), pd.DataFrame([[1, 2, 3]])]
)
def test_ensemble_append_unknown(ensemble, scm_run, obj):
    with pytest.raises(TypeError, match="Cannot handle appending type"):
        ensemble_append([obj])

    with pytest.raises(TypeError, match="Cannot handle appending type"):
        ensemble_append([scm_run, obj])


def test_ensemble_append_custom_run(custom_scm_run):
    res = ensemble_append([custom_scm_run, custom_scm_run])
    assert len(res) == len(custom_scm_run) * 2

    assert res.run_ids == [0, 1]

    # TODO: fix inconcistency with columns type of timeseries
    # pandas coerces this to a DataTimeIndex
    exp_ts = custom_scm_run.timeseries()
    exp_ts.columns = exp_ts.columns.astype(object)

    pdt.assert_frame_equal(res.timeseries().xs(0, level="run_id"), exp_ts)
    pdt.assert_frame_equal(res.timeseries().xs(1, level="run_id"), exp_ts)


@pytest.fixture("function")
def ensemble_unique_meta(scm_run):
    other_run = ScmRun(
        np.arange(3),
        index=[1900, 1910, 1920],
        columns={
            "variable": "Example",
            "unit": "unspecified",
            "region": "World",
            "model": "b_iam",
            "scenario": "a_scenario",
        },
    )
    return ScmEnsemble([scm_run, other_run])


def test_ensemble_unique_meta(ensemble_unique_meta):
    exp = ["Primary Energy", "Primary Energy|Coal", "Example"]
    assert ensemble_unique_meta.get_unique_meta("variable") == exp

    assert ensemble_unique_meta.get_unique_meta("model") == ["a_iam", "b_iam"]
    assert ensemble_unique_meta.get_unique_meta("climate_model") == ["a_model"]


def test_ensemble_unique_meta_missing(ensemble_unique_meta):
    res = ensemble_unique_meta.get_unique_meta("climate_model")
    assert res == ["a_model"]

    with pytest.raises(KeyError, match=re.escape("[non_existent] is not in metadata")):
        ensemble_unique_meta.get_unique_meta("non_existent")

    # Empty Ensembles also raise KeyError
    with pytest.raises(KeyError, match=re.escape("[test] is not in metadata")):
        ScmEnsemble([]).get_unique_meta("test")


def test_ensemble_unique_meta_no_duplicates(ensemble_unique_meta):
    assert ensemble_unique_meta.get_unique_meta("region", no_duplicates=True) == "World"

    with pytest.raises(ValueError, match="`variable` column is not unique"):
        ensemble_unique_meta.get_unique_meta("variable", no_duplicates=True)
