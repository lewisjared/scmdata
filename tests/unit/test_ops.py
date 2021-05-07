import datetime as dt
import re
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pint_pandas
import pytest
from openscm_units import unit_registry
from pint.errors import DimensionalityError

from scmdata.run import ScmRun
from scmdata.testing import _check_pandas_less_110, assert_scmdf_almost_equal

pint_pandas.PintType.ureg = unit_registry


def get_ts(data, index, **kwargs):
    return ScmRun(data=data, index=index, columns=kwargs)


def get_single_ts(
    data=[1, 2, 3],
    index=[1, 2, 3],
    variable="Emissions|CO2",
    scenario="scen",
    model="mod",
    unit="GtC / yr",
    region="World",
    **kwargs
):

    return get_ts(
        data=data,
        index=index,
        variable=variable,
        scenario=scenario,
        model=model,
        unit=unit,
        region=region,
        **kwargs
    )


def get_multiple_ts(
    data=np.array([[1, 2, 3], [10, 20, 30]]).T,
    index=[2020, 2030, 2040],
    variable=["Emissions|CO2", "Emissions|CH4"],
    scenario="scen",
    model="mod",
    unit=["GtC / yr", "MtCH4 / yr"],
    region="World",
    **kwargs
):
    return get_ts(
        data=data,
        index=index,
        variable=variable,
        scenario=scenario,
        model=model,
        unit=unit,
        region=region,
        **kwargs
    )


@pytest.fixture
def base_single_scmrun():
    return get_single_ts(variable="Emissions|CO2")


@pytest.fixture
def other_single_scmrun():
    return get_single_ts(data=[-1, 0, 5], variable="Emissions|CO2|Fossil")


@pytest.fixture
def base_multiple_scmrun():
    return get_multiple_ts(scenario="Scenario A")


@pytest.fixture
def other_multiple_scmrun():
    return get_multiple_ts(
        data=np.array([[-1, 0, 3.2], [11.1, 20, 32.3]]).T, scenario="Scenario B"
    )


def convert_to_pint_name(unit):
    return str(unit_registry(unit).units)


OPS_MARK = pytest.mark.parametrize("op", ("add", "subtract", "multiply", "divide"))


def perform_op(base, other, op, reset_index):
    base_ts = base.timeseries().reset_index(reset_index, drop=True)
    other_ts = other.timeseries().reset_index(reset_index, drop=True)

    if op == "add":
        exp_ts = base_ts + other_ts

    elif op == "subtract":
        exp_ts = base_ts - other_ts

    elif op == "divide":
        exp_ts = base_ts / other_ts

    elif op == "multiply":
        exp_ts = base_ts * other_ts

    else:
        raise NotImplementedError(op)

    return exp_ts.reset_index()


@patch("scmdata.ops.has_scipy", False)
def test_no_scipy(scm_run):
    with pytest.raises(
        ImportError, match="scipy is not installed. Run 'pip install scipy'"
    ):
        scm_run.integrate()


@OPS_MARK
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_single_timeseries(op, base_single_scmrun, other_single_scmrun):
    res = getattr(base_single_scmrun, op)(
        other_single_scmrun, op_cols={"variable": "Emissions|CO2|AFOLU"}
    )

    exp_ts = perform_op(base_single_scmrun, other_single_scmrun, op, "variable")
    exp_ts["variable"] = "Emissions|CO2|AFOLU"

    if op in ["add", "subtract"]:
        exp_ts["unit"] = "gigatC / a"

    elif op == "multiply":
        exp_ts["unit"] = "gigatC ** 2 / a ** 2"

    elif op == "divide":
        exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_multiple_timeseries(op, base_multiple_scmrun, other_multiple_scmrun):
    res = getattr(base_multiple_scmrun, op)(
        other_multiple_scmrun, op_cols={"scenario": "A to B"}
    )

    exp_ts = perform_op(base_multiple_scmrun, other_multiple_scmrun, op, "scenario")
    exp_ts["scenario"] = "A to B"

    if op in ["add", "subtract"]:
        exp_ts["unit"] = exp_ts["unit"].apply(convert_to_pint_name).values

    elif op == "multiply":
        exp_ts["unit"] = (
            exp_ts["unit"]
            .apply(lambda x: convert_to_pint_name("({})**2".format(x)))
            .values
        )

    elif op == "divide":
        exp_ts["unit"] = "dimensionless"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_missing_series_error():
    base = get_multiple_ts(region=["World|R5LAM", "World|R5REF"])
    other = get_multiple_ts(region=["World|R5LAM", "World|R5OECD"])

    error_msg = re.escape(
        "No equivalent in `other` for "
        "[('model', 'mod'), ('region', 'World|R5REF'), ('scenario', 'scen')]"
    )
    with pytest.raises(KeyError, match=error_msg):
        base.add(other, op_cols={"variable": "Warming plus Cumulative emissions CO2"})


def test_different_unit_error():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    error_msg = re.escape(
        "Cannot convert from 'kelvin' ([temperature]) to "
        "'gigatC' ([carbon] * [mass])"
    )
    with pytest.raises(DimensionalityError, match=error_msg):
        base.add(other, op_cols={"variable": "Warming plus Cumulative emissions CO2"})


def test_multiple_ops_cols():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    res = base.add(
        other,
        op_cols={
            "variable": "Warming plus Cumulative emissions CO2",
            "unit": "nonsense",
        },
    )

    exp_ts = perform_op(base, other, "add", ["variable", "unit"])
    exp_ts["variable"] = "Warming plus Cumulative emissions CO2"
    exp_ts["unit"] = "nonsense"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def test_warming_per_gt():
    base = get_single_ts(variable="Surface Temperature", unit="K")
    other = get_single_ts(variable="Cumulative Emissions|CO2", unit="GtC")

    res = base.divide(
        other, op_cols={"variable": "Warming per Cumulative emissions CO2"}
    )

    exp_ts = perform_op(base, other, "divide", ["variable", "unit"])
    exp_ts["variable"] = "Warming per Cumulative emissions CO2"
    exp_ts["unit"] = "kelvin / gigatC"

    exp = ScmRun(exp_ts)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


def perform_pint_op(base, pint_obj, op):
    base_ts = base.timeseries().T
    unit_level = base_ts.columns.names.index("unit")
    base_ts = base_ts.pint.quantify(level=unit_level)

    out = []
    for _, series in base_ts.iteritems():
        if op == "add":
            op_series = series + pint_obj

        elif op == "subtract":
            op_series = series - pint_obj

        elif op == "divide":
            op_series = series / pint_obj

        elif op == "divide_inverse":
            op_series = pint_obj / series

        elif op == "multiply":
            op_series = series * pint_obj

        elif op == "multiply_inverse":
            op_series = pint_obj * series

        else:
            raise NotImplementedError(op)

        out.append(op_series)

    out = pd.concat(out, axis="columns")
    out.columns.names = base_ts.columns.names
    out = out.pint.dequantify().T

    return out


@OPS_MARK
def test_scalar_ops_pint(op):
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_pint_op(start, scalar, op)
    exp = ScmRun(exp_ts)

    if op in ["add", "subtract"]:
        exp["unit"] = "gigatC / a"

    elif op == "multiply":
        exp["unit"] = "gigatC * megatC / a ** 2"

    elif op == "divide":
        exp["unit"] = "gigatC / megatC"

    if op == "add":
        res = start + scalar

    elif op == "subtract":
        res = start - scalar

    elif op == "divide":
        res = start / scalar

    elif op == "multiply":
        res = start * scalar

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.xfail(reason="pint doesn't recognise ScmRun")
def test_scalar_divide_pint_by_run():
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_pint_op(start, scalar, "divide_inverse")
    exp = ScmRun(exp_ts)

    exp["unit"] = "megatC / gigatC"

    res = scalar / start

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.xfail(reason="pint doesn't recognise ScmRun")
def test_scalar_multiply_pint_by_run():
    scalar = 1 * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_pint_op(start, scalar, "multiply_inverse")
    exp = ScmRun(exp_ts)

    exp["unit"] = "megatC * gigatC / a**2"

    res = scalar * start

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.parametrize("op", ["add", "subtract"])
def test_scalar_ops_pint_wrong_unit(op):
    scalar = 1 * unit_registry("Mt CH4 / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    error_msg = re.escape(
        "Cannot convert from 'gigatC / a' ([carbon] * [mass] / [time]) to 'CH4 * megametric_ton / a' ([mass] * [methane] / [time])"
    )
    with pytest.raises(DimensionalityError, match=error_msg):
        if op == "add":
            start + scalar

        elif op == "subtract":
            start - scalar

        else:
            raise NotImplementedError(op)


@OPS_MARK
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_vector_ops_pint(op):
    vector = np.arange(3) * unit_registry("MtC / yr")
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_pint_op(start, vector, op)
    exp = ScmRun(exp_ts)

    if op in ["add", "subtract"]:
        exp["unit"] = "gigatC / a"

    elif op == "multiply":
        exp["unit"] = "gigatC * megatC / a ** 2"

    elif op == "divide":
        exp["unit"] = "gigatC / megatC"

    if op == "add":
        res = start + vector

    elif op == "subtract":
        res = start - vector

    elif op == "divide":
        res = start / vector

    elif op == "multiply":
        res = start * vector

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@pytest.mark.parametrize("op", ["add", "subtract"])
@pytest.mark.parametrize("start_unit", ("GtC / yr", ["Mt CH4 / yr", "GtC / yr"]))
def test_vector_ops_pint_wrong_unit(op, start_unit):
    vector = np.arange(3) * unit_registry("Mt CH4 / yr")
    start = get_multiple_ts(
        variable="Emissions|Gas", unit=start_unit, scenario=["scen_a", "scen_b"]
    )

    error_msg = re.escape(
        "Cannot convert from 'gigatC / a' ([carbon] * [mass] / [time]) to 'CH4 * megametric_ton / a' ([mass] * [methane] / [time])"
    )
    with pytest.raises(DimensionalityError, match=error_msg):
        if op == "add":
            start + vector

        elif op == "subtract":
            start - vector

        else:
            raise NotImplementedError(op)


def perform_op_float_int(base, scalar, op):
    base_ts = base.timeseries()

    if op == "add":
        base_ts = base_ts + scalar

    elif op == "subtract":
        base_ts = base_ts - scalar

    elif op == "divide":
        base_ts = base_ts / scalar

    elif op == "multiply":
        base_ts = base_ts * scalar

    else:
        raise NotImplementedError(op)

    return base_ts


@OPS_MARK
@pytest.mark.parametrize("scalar", (1, 1.0))
def test_scalar_ops_float_int(op, scalar):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    exp_ts = perform_op_float_int(start, scalar, op)
    exp = ScmRun(exp_ts)

    if op == "add":
        res = start + scalar

    elif op == "subtract":
        res = start - scalar

    elif op == "divide":
        res = start / scalar

    elif op == "multiply":
        res = start * scalar

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
@pytest.mark.parametrize("shape", ((2, 2), (3, 2), (3, 3, 3)))
def test_wrong_shape_ops(op, shape):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    other = np.arange(np.prod(shape)).reshape(shape)

    error_msg = re.escape(
        "operations with {}d data are not supported".format(len(shape))
    )
    with pytest.raises(ValueError, match=error_msg):
        if op == "add":
            start + other

        elif op == "subtract":
            start - other

        elif op == "divide":
            start / other

        elif op == "multiply":
            start * other

        else:
            raise NotImplementedError(op)


@OPS_MARK
@pytest.mark.parametrize(
    "vector", (np.arange(3).astype(int), np.arange(3).astype(float))
)
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_vector_ops_float_int(op, vector):
    start = get_multiple_ts(
        variable="Emissions|Gas",
        unit=["GtC / yr", "Mt CH4 / yr"],
        scenario=["scen_a", "scen_b"],
    )

    exp_ts = perform_op_float_int(start, vector, op)
    exp = ScmRun(exp_ts)

    if op == "add":
        res = start + vector

    elif op == "subtract":
        res = start - vector

    elif op == "divide":
        res = start / vector

    elif op == "multiply":
        res = start * vector

    else:
        raise NotImplementedError(op)

    assert_scmdf_almost_equal(res, exp, allow_unordered=True, check_ts_names=False)


@OPS_MARK
def test_wrong_length_ops(op):
    start = get_multiple_ts(
        variable="Emissions|CO2", unit="GtC / yr", scenario=["scen_a", "scen_b"]
    )

    other = np.arange(start.shape[1] - 1)

    error_msg = re.escape(
        "only vectors with the same number of timesteps as self (3) are supported"
    )
    with pytest.raises(ValueError, match=error_msg):
        if op == "add":
            start + other

        elif op == "subtract":
            start - other

        elif op == "divide":
            start / other

        elif op == "multiply":
            start * other

        else:
            raise NotImplementedError(op)


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
@pytest.mark.parametrize("out_var", (None, "new out var"))
# We can add initial back if use case arises. At the moment I can't see an easy
# way to make the units behave.
# @pytest.mark.parametrize("initial", (None, 0, 1, -1.345))
def test_integration(out_var):
    dat = [1, 2, 3]
    start = get_single_ts(data=dat, index=[1, 2, 3], unit="GtC / yr")

    res = start.integrate(out_var=out_var)

    if out_var is None:
        exp_var = ("Cumulative " + start["variable"]).values
    else:
        exp_var = out_var

    exp = get_single_ts(
        data=np.array([0, 1.5, 4]), index=[1, 2, 3], variable=exp_var, unit="gigatC"
    )
    # rtol is because our calculation uses seconds, which doesn't work out
    # quite the same as assuming a regular year
    assert_scmdf_almost_equal(
        res, exp, allow_unordered=True, check_ts_names=False, rtol=1e-3
    )


def test_integration_time_handling_big_jumps():
    start = get_single_ts(data=[1, 2, 3], index=[10, 20, 50], unit="GtC / yr")

    res = start.integrate()

    npt.assert_allclose(
        res.values.squeeze(), [0, 15, 90], rtol=1e-3,
    )


def test_integration_time_handling_all_over_jumps():
    start = get_single_ts(
        data=[1, 2, 3, 3, 1.8], index=[10, 10.1, 11, 20, 50], unit="GtC / yr"
    )

    res = start.integrate()

    first = 0
    second = first + 1.5 * 0.1
    third = second + 2.5 * 0.9
    fourth = third + 3 * 9
    fifth = fourth + 2.4 * 30
    npt.assert_allclose(
        res.values.squeeze(), [first, second, third, fourth, fifth], rtol=1e-3
    )


def test_integration_nan_handling():
    start = get_single_ts(
        data=[1, 2, 3, np.nan, 12, np.nan, 30, 40],
        index=[10, 20, 50, 60, 70, 80, 90, 100],
        unit="GtC / yr",
    )

    warn_msg = re.escape(
        "You are integrating data which contains nans so your result will also "
        "contain nans. Perhaps you want to remove the nans before performing "
        "the integration using a combination of :meth:`filter` and "
        ":meth:`interpolate`?"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        res = start.integrate()

    npt.assert_allclose(
        res.values.squeeze(),
        [0, 15, 90, np.nan, np.nan, np.nan, np.nan, np.nan],
        rtol=1e-3,
    )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
def test_integration_multiple_ts():
    variables = ["Emissions|CO2", "Heat Uptake", "Temperature"]
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 5, 10]]).T,
        index=[2020, 2025, 2040],
        variable=variables,
        unit=["Mt CO2 / yr", "W / m^2", "K"],
    )

    res = start.integrate()

    exp = get_single_ts(
        data=np.array([[0, 7.5, 45], [0, -7.5, -45], [0, 12.5, 125]]).T,
        index=[2020, 2025, 2040],
        variable=["Cumulative {}".format(v) for v in variables],
        unit=["Mt CO2", "W / m^2 * yr", "K * yr"],
    )

    for v in variables:
        cv = "Cumulative {}".format(v)
        exp_comp = exp.filter(variable=cv)
        res_comp = res.filter(variable=cv).convert_unit(
            exp_comp.get_unique_meta("unit", no_duplicates=True),
        )

        assert_scmdf_almost_equal(
            res_comp, exp_comp, allow_unordered=True, check_ts_names=False, rtol=1e-3
        )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
@pytest.mark.parametrize("out_var", (None, "new out var"))
def test_delta_per_delta_time(out_var):
    dat = [1, 2, 3]
    start = get_single_ts(data=dat, index=[1, 2, 3], unit="GtC / yr")

    res = start.delta_per_delta_time(out_var=out_var).convert_unit("GtC / yr^2")

    if out_var is None:
        exp_var = ("Delta " + start["variable"]).values
    else:
        exp_var = out_var

    exp = get_single_ts(
        data=np.array([1, 1]), index=[1.5, 2.5], variable=exp_var, unit="GtC / yr^2"
    )
    # rtol is because our calculation uses seconds, which doesn't work out
    # quite the same as assuming a regular year
    assert_scmdf_almost_equal(
        res, exp, allow_unordered=True, check_ts_names=False, rtol=1e-3
    )


def test_delta_per_delta_time_handling_big_jumps():
    start = get_single_ts(data=[1, 2, 3], index=[10, 20, 50], unit="GtC")

    res = start.delta_per_delta_time().convert_unit("GtC / yr")

    npt.assert_allclose(
        res.values.squeeze(), [1 / 10, 1 / 30], rtol=1e-3,
    )


def test_delta_per_delta_time_handling_all_over_jumps():
    start = get_single_ts(
        data=[1, 2, 3, 3, 1.8], index=[10, 10.1, 11, 20, 50], unit="GtC"
    )

    res = start.delta_per_delta_time().convert_unit("GtC / yr")

    npt.assert_allclose(res.values.squeeze(), [10, 1 / 0.9, 0, -1.2 / 30], rtol=1e-3)


def test_delta_per_delta_time_nan_handling():
    start = get_single_ts(
        data=[1, 2, 3, np.nan, 12, np.nan, 30, 40],
        index=[10, 20, 50, 60, 70, 80, 90, 100],
        unit="GtC",
    )

    warn_msg = re.escape(
        "You are calculating deltas of data which contains nans so your result "
        "will also contain nans. Perhaps you want to remove the nans before "
        "calculating the deltas using a combination of :meth:`filter` and "
        ":meth:`interpolate`?"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        res = start.delta_per_delta_time().convert_unit("GtC / yr")

    npt.assert_allclose(
        res.values.squeeze(),
        [1 / 10, 1 / 30, np.nan, np.nan, np.nan, np.nan, 1],
        rtol=1e-3,
    )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
def test_delta_per_delta_time_multiple_ts():
    variables = ["Emissions|CO2", "Heat Uptake", "Temperature"]
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 5, 10]]).T,
        index=[2020, 2025, 2040],
        variable=variables,
        unit=["Mt CO2", "J / m^2", "K"],
    )

    res = start.delta_per_delta_time()

    assert sorted(res["unit"]) == sorted(
        [
            "CO2 * megametric_ton / second",
            "joule / meter ** 2 / second",
            "kelvin / second",
        ]
    )

    res = (
        res.convert_unit("Mt CO2 / yr", variable="Delta Emissions|CO2")
        .convert_unit("J / m^2 / yr", variable="Delta Heat Uptake")
        .convert_unit("K / yr", variable="Delta Temperature")
    )

    exp = get_single_ts(
        data=np.array([[1 / 5, 1 / 15], [-1 / 5, -1 / 15], [5 / 5, 5 / 15]]).T,
        index=[2022.5, (2025 + 2040) / 2],
        variable=["Delta {}".format(v) for v in variables],
        unit=["Mt CO2 / yr", "J / m^2 / yr", "K / yr"],
    )

    for v in variables:
        cv = "Delta {}".format(v)
        exp_comp = exp.filter(variable=cv)
        res_comp = res.filter(variable=cv).convert_unit(
            exp_comp.get_unique_meta("unit", no_duplicates=True),
        )

        assert_scmdf_almost_equal(
            res_comp, exp_comp, allow_unordered=True, check_ts_names=False, rtol=1e-3
        )


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
def test_linear_regression():
    dat = [1, 2, 3]
    start = get_single_ts(data=dat, index=[1970, 1971, 1972], unit="GtC / yr")

    res = start.linear_regression()

    assert len(res) == 1
    assert res[0]["variable"] == "Emissions|CO2"
    assert res[0]["scenario"] == "scen"
    assert res[0]["model"] == "mod"
    assert res[0]["region"] == "World"
    npt.assert_allclose(res[0]["gradient"].to("GtC / yr / yr").magnitude, 1, rtol=1e-3)
    npt.assert_allclose(res[0]["intercept"].to("GtC / yr").magnitude, 1, rtol=1e-3)


def test_linear_regression_handling_big_jumps():
    start = get_single_ts(data=[1, 2, 3], index=[10, 20, 50], unit="GtC")

    res = start.linear_regression()

    npt.assert_allclose(res[0]["gradient"].to("GtC / yr").magnitude, 0.04615, rtol=1e-3)


def test_linear_regression_handling_all_over_jumps():
    start = get_single_ts(
        data=[1, 2, 3, 3, 1.8], index=[10, 10.1, 11, 20, 50], unit="GtC"
    )

    res = start.linear_regression()

    npt.assert_allclose(
        res[0]["gradient"].to("GtC / yr").magnitude, -0.00439, rtol=1e-3
    )


def test_linear_regression_nan_handling():
    start = get_single_ts(
        data=[1, 2, 3, np.nan, 12, np.nan, 30, 40],
        index=[10, 20, 50, 60, 70, 80, 90, 100],
        unit="GtC",
    )

    warn_msg = re.escape(
        "You are calculating a linear regression of data which contains nans so your result "
        "will also contain nans. Perhaps you want to remove the nans before "
        "calculating the regression using a combination of :meth:`filter` and "
        ":meth:`interpolate`?"
    )
    with pytest.warns(UserWarning, match=warn_msg):
        res = start.linear_regression()

    assert np.isnan(res[0]["gradient"])
    assert np.isnan(res[0]["intercept"])


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
def test_linear_regression_multiple_ts():
    variables = ["Emissions|CO2", "Heat Uptake", "Temperature", "Temperature Ocean"]
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 5, 10], [1, 4, 8]]).T,
        index=[2020, 2021, 2022],
        variable=variables,
        unit=["Mt CO2", "J / m^2", "K", "K"],
    )

    res = start.linear_regression()

    assert len(res) == 4
    for r in res:
        if r["variable"] == "Emissions|CO2":
            npt.assert_allclose(r["gradient"].to("Mt CO2 / yr").magnitude, 1, rtol=1e-3)
        elif r["variable"] == "Heat Uptake":
            npt.assert_allclose(
                r["gradient"].to("J / m^2 / yr").magnitude, -1, rtol=1e-3
            )
        elif r["variable"] == "Temperature":
            npt.assert_allclose(r["gradient"].to("K / yr").magnitude, 5, rtol=1e-3)
        elif r["variable"] == "Temperature Ocean":
            npt.assert_allclose(r["gradient"].to("K / yr").magnitude, 3.5, rtol=1e-3)
        else:
            raise NotImplementedError(r["variable"])


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
@pytest.mark.parametrize(
    "unit,exp_values",
    (
        ("Mt CO2 / yr", [1, -1, 5, 5 * 10 ** 3 * 44 / 12]),
        ("Mt CO2 / day", np.array([1, -1, 5, 5 * 10 ** 3 * 44 / 12]) / 365.25),
        (None, np.array([1, -1, 5, 5]) / (365.25 * 24 * 60 * 60)),
    ),
)
def test_linear_regression_gradient(unit, exp_values):
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 5, 10], [0, 5, 10]]).T,
        index=[2020, 2021, 2022],
        variable="Emissions|CO2",
        unit=["Mt CO2", "Mt CO2", "Mt CO2", "GtC"],
        scenario=["a", "b", "c", "d"],
    )

    res = start.linear_regression_gradient(unit=unit)

    exp = start.meta
    exp["gradient"] = exp_values
    exp["unit"] = (
        unit
        if unit is not None
        else [
            "CO2 * megametric_ton / second",
            "CO2 * megametric_ton / second",
            "CO2 * megametric_ton / second",
            "gigatC / second",
        ]
    )

    pdt.assert_frame_equal(res, exp, rtol=1e-3, check_like=True)


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
@pytest.mark.parametrize(
    "unit,exp_values",
    (
        ("Mt CO2", [2, -2, 6, 5 * 10 ** 3 * 44 / 12]),
        (
            "GtC",
            [
                2 / (10 ** 3) * 12 / 44,
                -2 / (10 ** 3) * 12 / 44,
                6 / (10 ** 3) * 12 / 44,
                5,
            ],
        ),
        (None, [2, -2, 6, 5]),
    ),
)
def test_linear_regression_intercept(unit, exp_values):
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 8, 10], [0, 5, 10]]).T,
        index=[1969, 1970, 1971],
        variable="Emissions|CO2",
        unit=["Mt CO2", "Mt CO2", "Mt CO2", "GtC"],
        scenario=["a", "b", "c", "d"],
    )

    res = start.linear_regression_intercept(unit=unit)

    exp = start.meta
    exp["intercept"] = np.array(exp_values).astype(float)
    exp["unit"] = (
        unit
        if unit is not None
        else [
            "CO2 * megametric_ton / second",
            "CO2 * megametric_ton / second",
            "CO2 * megametric_ton / second",
            "gigatC / second",
        ]
    )

    pdt.assert_frame_equal(res, exp, rtol=1e-3, check_like=True)


@pytest.mark.xfail(
    _check_pandas_less_110(), reason="pandas<=1.1.0 does not have rtol argument"
)
def test_linear_regression_scmrun():
    start = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [0, 8, 10], [0, 5, 10]]).T,
        index=[1969, 1970, 1971],
        variable="Emissions|CO2",
        unit=["Mt CO2 / yr", "Mt CO2 / yr", "Mt CO2 / yr", "GtC / yr"],
        scenario=["a", "b", "c", "d"],
    )

    res = start.linear_regression_scmrun()

    exp = get_multiple_ts(
        data=np.array([[1, 2, 3], [-1, -2, -3], [1, 6, 11], [0, 5, 10]]).T,
        index=[1969, 1970, 1971],
        variable="Emissions|CO2",
        unit=["Mt CO2 / yr", "Mt CO2 / yr", "Mt CO2 / yr", "GtC / yr"],
        scenario=["a", "b", "c", "d"],
    )

    assert_scmdf_almost_equal(
        res, exp, allow_unordered=True, check_ts_names=False, rtol=1e-3
    )


# TODO: notebook illustrating rolling mean options
# Rationale: rolling means are really tricky (do you take e.g. an annual mean first, do you worry about happens at the window edge?) and they're pretty easy to convert back into ScmRun objects so a notebooks is probably more helpful than exact functionality for now
