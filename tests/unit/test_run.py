import copy
import datetime as dt
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas.errors import UnsupportedFunctionCall
from pint.errors import DimensionalityError, UndefinedUnitError

from scmdata.dataframe import ScmDataFrame
from scmdata.run import ScmRun, df_append
from scmdata.testing import assert_scmdf_almost_equal


def test_init_df_year_converted_to_datetime(test_pd_df):
    res = ScmRun(test_pd_df)
    assert (res["year"].unique() == [2005, 2010, 2015]).all()
    assert (
        res["time"].unique()
        == [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)]
    ).all()


@pytest.mark.parametrize(
    "in_format",
    [
        "pd.Series",
        "year_col",
        "year_col_index",
        "time_col",
        "time_col_index",
        "time_col_str_simple",
        "time_col_str_complex",
        "str_times",
    ],
)
def test_init_df_formats(test_pd_run_df, in_format):
    if in_format == "pd.Series":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year").set_index(
            idx + ["year"]
        )["value"]
    elif in_format == "year_col":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
    elif in_format == "year_col_index":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year").set_index(
            idx + ["year"]
        )
    elif in_format == "time_col":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: dt.datetime(x, 1, 1))
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "time_col_index":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: dt.datetime(x, 1, 1))
        test_init = test_init.drop("year", axis="columns")
        test_init = test_init.set_index(idx + ["time"])
    elif in_format == "time_col_str_simple":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(
            lambda x: "{}-1-1 00:00:00".format(x)
        )
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "time_col_str_complex":
        idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
        test_init = test_pd_run_df.melt(id_vars=idx, var_name="year")
        test_init["time"] = test_init["year"].apply(lambda x: "{}/1/1".format(x))
        test_init = test_init.drop("year", axis="columns")
    elif in_format == "str_times":
        test_init = test_pd_run_df.copy()
        test_init.columns = test_init.columns.map(
            lambda x: "{}/1/1".format(x) if isinstance(x, int) else x
        )

    res = ScmRun(test_init)
    assert (res["year"].unique() == [2005, 2010, 2015]).all()
    assert (
        res["time"].unique()
        == [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)]
    ).all()

    res_df = res.timeseries()
    res_df.columns = res_df.columns.map(lambda x: x.year)
    res_df = res_df.reset_index()

    pd.testing.assert_frame_equal(
        res_df[test_pd_run_df.columns.tolist()], test_pd_run_df, check_like=True
    )


def test_init_df_missing_time_axis_error(test_pd_df):
    idx = ["climate_model", "model", "scenario", "region", "variable", "unit"]
    test_init = test_pd_df.melt(id_vars=idx, var_name="year")
    test_init = test_init.drop("year", axis="columns")
    error_msg = re.escape("invalid time format, must have either `year` or `time`!")
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(test_init)


def test_init_df_missing_time_columns_error(test_pd_df):
    test_init = test_pd_df.copy()
    test_init = test_init.drop(
        test_init.columns[test_init.columns.map(lambda x: isinstance(x, int))],
        axis="columns",
    )
    error_msg = re.escape(
        "invalid column format, must contain some time (int, float or datetime) columns!"
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(test_init)


def test_init_df_missing_col_error(test_pd_df):
    test_pd_df = test_pd_df.drop("model", axis="columns")
    error_msg = re.escape("missing required columns `['model']`!")
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(test_pd_df)


def test_init_ts_missing_col_error(test_ts):
    error_msg = re.escape("missing required columns `['model']`!")
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(
            test_ts,
            columns={
                "climate_model": ["a_model"],
                "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            index=[2005, 2010, 2015],
        )


def test_init_multiple_file_error():
    error_msg = re.escape(
        "Initialising from multiple files not supported, use "
        "`scmdata.dataframe.ScmDataFrame.append()`"
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(["file_1", "filepath_2"])


def test_init_unrecognised_type_error():
    fail_type = {"dict": "key"}
    error_msg = re.escape(
        "Cannot load <class 'scmdata.dataframe.ScmDataFrame'> from {}".format(
            type(fail_type)
        )
    )
    with pytest.raises(TypeError, match=error_msg):
        ScmDataFrame(fail_type)


def test_init_ts_col_string(test_ts):
    res = ScmDataFrame(
        test_ts,
        columns={
            "model": "an_iam",
            "climate_model": "a_model",
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": "World",
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": "EJ/yr",
        },
        index=[2005, 2010, 2015],
    )
    npt.assert_array_equal(res["model"].unique(), "an_iam")
    npt.assert_array_equal(res["climate_model"].unique(), "a_model")
    npt.assert_array_equal(res["region"].unique(), "World")
    npt.assert_array_equal(res["unit"].unique(), "EJ/yr")


@pytest.mark.parametrize("fail_setting", [["a_iam", "a_iam"]])
def test_init_ts_col_wrong_length_error(test_ts, fail_setting):
    correct_scenarios = ["a_scenario", "a_scenario", "a_scenario2"]
    error_msg = re.escape(
        "Length of column 'model' is incorrect. It should be length 1 or {}".format(
            len(correct_scenarios)
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(
            test_ts,
            columns={
                "model": fail_setting,
                "climate_model": ["a_model"],
                "scenario": correct_scenarios,
                "region": ["World"],
                "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
                "unit": ["EJ/yr"],
            },
            index=[2005, 2010, 2015],
        )


def get_test_pd_df_with_datetime_columns(tpdf):
    return tpdf.rename(
        {
            2005.0: dt.datetime(2005, 1, 1),
            2010.0: dt.datetime(2010, 1, 1),
            2015.0: dt.datetime(2015, 1, 1),
        },
        axis="columns",
    )


def test_init_ts(test_ts, test_pd_df):
    df = ScmDataFrame(
        test_ts,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": ["EJ/yr"],
        },
        index=[2005, 2010, 2015],
    )

    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)

    b = ScmDataFrame(test_pd_df)

    assert_scmdf_almost_equal(df, b, allow_unordered=True)


@pytest.mark.parametrize(
    "years", [["2005.0", "2010.0", "2015.0"], ["2005", "2010", "2015"]]
)
def test_init_with_years_as_str(test_pd_df, years):
    df = copy.deepcopy(
        test_pd_df
    )  # This needs to be a deep copy so it doesn't break the other tests
    cols = copy.deepcopy(test_pd_df.columns.values)
    cols[-3:] = years
    df.columns = cols

    df = ScmDataFrame(df)

    obs = df._data.index
    exp = pd.Index(
        [dt.datetime(2005, 1, 1), dt.datetime(2010, 1, 1), dt.datetime(2015, 1, 1)],
        name="time",
        dtype="object",
    )
    pd.testing.assert_index_equal(obs, exp)


def test_init_with_year_columns(test_pd_df):
    df = ScmDataFrame(test_pd_df)
    tdf = get_test_pd_df_with_datetime_columns(test_pd_df)
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_with_decimal_years():
    inp_array = [2.0, 1.2, 7.9]
    d = pd.Series(inp_array, index=[1765.0, 1765.083, 1765.167])
    cols = {
        "model": ["a_model"],
        "scenario": ["a_scenario"],
        "region": ["World"],
        "variable": ["Primary Energy"],
        "unit": ["EJ/yr"],
    }

    res = ScmDataFrame(d, columns=cols)
    assert (
        res["time"].unique()
        == [
            dt.datetime(1765, 1, 1, 0, 0),
            dt.datetime(1765, 1, 31, 7, 4, 48),
            dt.datetime(1765, 3, 2, 22, 55, 11),
        ]
    ).all()
    npt.assert_array_equal(res._data.loc[:, 0].values, inp_array)


def test_init_df_from_timeseries(test_scm_run_mulitple):
    df = ScmRun(test_scm_run_mulitple.timeseries())

    assert_scmdf_almost_equal(df, test_scm_run_mulitple, check_ts_names=False)


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "test value"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = ScmDataFrame(tdf)

    tdf = get_test_pd_df_with_datetime_columns(tdf)
    assert extra_col in df.meta
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_iam(test_iam_df, test_pd_df):
    a = ScmDataFrame(test_iam_df)
    b = ScmDataFrame(test_pd_df)

    assert_scmdf_almost_equal(a, b)


def test_init_self(test_iam_df):
    a = ScmDataFrame(test_iam_df)
    b = ScmDataFrame(a)

    assert_scmdf_almost_equal(a, b)


def test_as_iam(test_iam_df, test_pd_df, iamdf_type):
    df = ScmDataFrame(test_pd_df).to_iamdataframe()

    # test is skipped by test_iam_df fixture if pyam isn't installed
    assert isinstance(df, iamdf_type)

    pd.testing.assert_frame_equal(test_iam_df.meta, df.meta)
    # we switch to time so ensure sensible comparison of columns
    tdf = df.data.copy()
    tdf["year"] = tdf["time"].apply(lambda x: x.year)
    tdf.drop("time", axis="columns", inplace=True)
    pd.testing.assert_frame_equal(test_iam_df.data, tdf, check_like=True)


def test_get_item(test_scm_run):
    assert test_scm_run["model"].unique() == ["a_iam"]


def test_get_item_not_in_meta(test_scm_run):
    dud_key = 0
    error_msg = re.escape("I don't know what to do with key: {}".format(dud_key))
    with pytest.raises(KeyError, match=error_msg):
        test_scm_run[dud_key]


def test_set_item(test_scm_run):
    test_scm_run["model"] = ["a_iam", "b_iam", "c_iam"]
    assert all(test_scm_run["model"] == ["a_iam", "b_iam", "c_iam"])


def test_set_item_not_in_meta(test_scm_run):
    with pytest.raises(ValueError):
        test_scm_run["junk"] = ["hi", "bye"]

    test_scm_run["junk"] = ["hi", "bye", "ciao"]
    assert all(test_scm_run["junk"] == ["hi", "bye", "ciao"])


def test_len(test_scm_run):
    assert len(test_scm_run) == len(test_scm_run._ts)


def test_head(test_scm_run):
    pd.testing.assert_frame_equal(test_scm_run.head(2), test_scm_run.timeseries().head(2))


def test_tail(test_scm_run):
    pd.testing.assert_frame_equal(test_scm_run.tail(1), test_scm_run.timeseries().tail(1))


def test_values(test_scm_run):
    npt.assert_array_equal(test_scm_run.values, test_scm_run.timeseries().values)


def test_variable_depth_0(test_scm_run):
    obs = list(test_scm_run.filter(level=0)["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_with_base():
    tdf = ScmDataFrame(
        data=np.array([[1, 6.0, 7], [0.5, 3, 2], [2, 7, 0], [-1, -2, 3]]).T,
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario"],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy|Coal|Electricity",
                "Primary Energy|Gas|Heating",
            ],
            "unit": ["EJ/yr"],
        },
        index=[
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )

    obs = list(tdf.filter(variable="Primary Energy|*", level=1)["variable"].unique())
    exp = ["Primary Energy|Coal|Electricity", "Primary Energy|Gas|Heating"]
    assert all([e in obs for e in exp]) and len(obs) == len(exp)

@pytest.mark.xfail
def test_variable_depth_0_keep_false(test_scm_run):
    obs = list(test_scm_run.filter(level=0, keep=False)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_0_minus(test_scm_run):
    obs = list(test_scm_run.filter(level="0-")["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_plus(test_scm_run):
    obs = list(test_scm_run.filter(level="0+")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1(test_scm_run):
    obs = list(test_scm_run.filter(level=1)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_minus(test_scm_run):
    obs = list(test_scm_run.filter(level="1-")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_plus(test_scm_run):
    obs = list(test_scm_run.filter(level="1+")["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_raises(test_scm_run):
    pytest.raises(ValueError, test_scm_run.filter, level="1/")


def test_filter_error(test_scm_run):
    pytest.raises(ValueError, test_scm_run.filter, foo="foo")


def test_filter_year(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(year=2005)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_year_error(test_scm_datetime_df):
    error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(year=2005.0)


def test_filter_inplace(test_scm_datetime_df):
    test_scm_datetime_df.filter(year=2005, inplace=True)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = test_scm_datetime_df["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_scm_datetime_df, test_month):
    obs = test_scm_datetime_df.filter(month=test_month)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_scm_datetime_df, test_month):
    obs = test_scm_datetime_df.filter(year=2005, month=test_month)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_scm_datetime_df, test_day):
    obs = test_scm_datetime_df.filter(day=test_day)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("test_hour", [12, [12, 13]])
def test_filter_hour(test_scm_datetime_df, test_hour):
    obs = test_scm_datetime_df.filter(hour=test_hour)
    test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
    expected_rows = test_scm_datetime_df["time"].apply(lambda x: x.hour).isin(test_hour)
    expected = test_scm_datetime_df["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_hour_multiple(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(hour=0)
    expected_rows = test_scm_datetime_df["time"].apply(lambda x: x.hour).isin([0])
    expected = test_scm_datetime_df["time"].loc[expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 2
    assert all([dt in unique_time for dt in expected])


def test_filter_time_exact_match(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(time=dt.datetime(2005, 6, 17, 12))
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range(test_scm_datetime_df):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(
            year=range(dt.datetime(2000, 6, 17), dt.datetime(2009, 6, 17))
        )


def test_filter_time_range_year(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(year=range(2000, 2008))

    unique_time = obs["time"].unique()
    expected = dt.datetime(2005, 6, 17, 12)

    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [range(3, 7), "Mar-Jun"])
def test_filter_time_range_month(test_scm_datetime_df, month_range):
    obs = test_scm_datetime_df.filter(month=month_range)
    expected = dt.datetime(2005, 6, 17, 12)

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range_month_unrecognised_error(test_scm_datetime_df):
    fail_filter = "Marb-Jun"
    error_msg = re.escape(
        "Could not convert month '{}' to integer".format(
            [m for m in fail_filter.split("-")]
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_datetime_df.filter(month=fail_filter)


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_scm_datetime_df, month_range):
    error_msg = re.escape(
        "string ranges must lead to increasing integer ranges, "
        "Nov-Feb becomes [11, 2]"
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_datetime_df.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_scm_datetime_df, day_range):
    obs = test_scm_datetime_df.filter(day=day_range)
    expected = dt.datetime(2005, 6, 17, 12)
    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected


def test_filter_time_range_day_unrecognised_error(test_scm_datetime_df):
    fail_filter = "Thud-Sat"
    error_msg = re.escape(
        "Could not convert day '{}' to integer".format(
            [m for m in fail_filter.split("-")]
        )
    )
    with pytest.raises(ValueError, match=error_msg):
        test_scm_datetime_df.filter(day=fail_filter)


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_scm_datetime_df, hour_range):
    obs = test_scm_datetime_df.filter(hour=hour_range)

    expected_rows = (
        test_scm_datetime_df["time"].apply(lambda x: x.hour).isin(hour_range)
    )
    expected = test_scm_datetime_df["time"][expected_rows].unique()

    unique_time = obs["time"].unique()
    assert len(unique_time) == 1
    assert unique_time[0] == expected[0]


def test_filter_time_no_match(test_scm_datetime_df):
    obs = test_scm_datetime_df.filter(time=dt.datetime(2004, 6, 18))
    assert len(obs.time_points) == 0
    assert obs.shape[1] == 0
    assert obs.values.shape[1] == 0


def test_filter_time_not_datetime_error(test_scm_datetime_df):
    error_msg = re.escape("`time` can only be filtered with datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(time=2005)


def test_filter_time_not_datetime_range_error(test_scm_datetime_df):
    error_msg = re.escape("`time` can only be filtered with datetimes")
    with pytest.raises(TypeError, match=error_msg):
        test_scm_datetime_df.filter(time=range(2000, 2008))


def test_filter_as_kwarg(test_scm_run):
    obs = list(test_scm_run.filter(variable="Primary Energy|Coal")["scenario"].unique())
    assert obs == ["a_scenario"]

@pytest.mark.xfail
def test_filter_keep_false(test_scm_run):
    df = test_scm_run.filter(variable="Primary Energy|Coal", year=2005, keep=False)
    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [1, 6, 6, np.nan, 3, 3])


def test_filter_by_regexp(test_scm_run):
    obs = test_scm_run.filter(scenario="a_scenari.$", regexp=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_timeseries_different_length():
    df = ScmDataFrame(
        pd.DataFrame(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]).T, index=[2000, 2001, 2002]
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/yr"],
        },
    )

    npt.assert_array_equal(
        df.filter(scenario="a_scenario2").timeseries().squeeze(), [4.0, 5.0]
    )
    npt.assert_array_equal(df.filter(year=2002).timeseries().squeeze(), 3.0)

    exp = pd.Series(["a_scenario"], name="scenario")
    obs = df.filter(year=2002)["scenario"]
    pd.testing.assert_series_equal(exp, obs)
    assert df.filter(scenario="a_scenario2", year=2002).timeseries().empty


@pytest.mark.parametrize("has_nan", [True, False])
def test_filter_timeseries_nan_meta(has_nan):
    df = ScmDataFrame(
        pd.DataFrame(
            np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]).T, index=[2000, 2001]
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario2", np.nan],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/yr"],
        },
    )

    # not sure how we want to setup NaN filtering, empty string seems as good as any?
    if not has_nan:
        error_msg = re.escape(
            "String filtering cannot be performed on column 'scenario', which "
            "contains NaN's, unless `has_nan` is True"
        )
        with pytest.raises(TypeError, match=error_msg):
            df.filter(scenario="*", has_nan=has_nan)
        with pytest.raises(TypeError, match=error_msg):
            df.filter(scenario="", has_nan=has_nan)

    else:

        def with_nan_assertion(a, b):
            assert all(
                [
                    (v == b[i]) or (np.isnan(v) and np.isnan(b[i]))
                    for i, v in enumerate(a)
                ]
            )

        res = df.filter(scenario="*", has_nan=has_nan)["scenario"].unique()
        exp = ["a_scenario", "a_scenario2", np.nan]
        with_nan_assertion(res, exp)

        res = df.filter(scenario="", has_nan=has_nan)["scenario"].unique()
        exp = [np.nan]
        with_nan_assertion(res, exp)


def test_timeseries(test_scm_run):
    dct = {
        "model": ["a_model"] * 3,
        "scenario": ["a_scenario"] * 3,
        "years": [2005, 2010, 2015],
        "value": [1, 6, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    obs = test_scm_run.filter(
        variable="Primary Energy", scenario="a_scenario"
    ).timeseries()
    npt.assert_array_equal(obs, exp)


def test_timeseries_meta(test_scm_run):
    obs = test_scm_run.filter(variable="Primary Energy").timeseries(
        meta=["scenario", "model"]
    )
    npt.assert_array_equal(obs.index.names, ["scenario", "model"])


def test_timeseries_duplicated(test_scm_run):
    pytest.raises(ValueError, test_scm_run.timeseries, meta=["scenario"])


def test_quantile_over_lower(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/yr", -1.0, -2.0, 0.0],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "quantile", q=0)
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_quantile_over_upper(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "World", "Primary Energy", "EJ/yr", 2.0, 7.0, 7.0],
            ["a_model", "World", "Primary Energy|Coal", "EJ/yr", 0.5, 3.0, 2.0],
        ],
        columns=[
            "climate_model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over(["model", "scenario"], "quantile", q=1)
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_mean_over(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy",
                "EJ/yr",
                2 / 3,
                11 / 3,
                10 / 3,
            ],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "mean")
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_median_over(test_processing_scm_df):
    exp = pd.DataFrame(
        [
            ["a_model", "a_iam", "World", "Primary Energy", "EJ/yr", 1.0, 6.0, 3.0],
            [
                "a_model",
                "a_iam",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3.0,
                2.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )
    obs = test_processing_scm_df.process_over("scenario", "median")
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_process_over_unrecognised_operation_error(test_scm_run):
    error_msg = re.escape("operation must be one of ['median', 'mean', 'quantile']")
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run.process_over("scenario", "junk")


def test_process_over_kwargs_error(test_scm_run):
    with pytest.raises(UnsupportedFunctionCall):
        test_scm_run.process_over("scenario", "mean", junk=4)


@pytest.mark.parametrize(
    "tfilter,tappend_str,exp_append_str",
    [
        (
            {"time": [dt.datetime(y, 1, 1, 0, 0, 0) for y in range(2005, 2011)]},
            None,
            "(ref. period time: 2005-01-01 00:00:00 - 2010-01-01 00:00:00)",
        ),
        ({"month": [1, 2, 3]}, "(Jan - Mar)", "(Jan - Mar)"),
        ({"day": [1, 2, 3]}, None, "(ref. period day: 1 - 3)"),
    ],
)
def test_relative_to_ref_period_mean(
    test_processing_scm_df, tfilter, tappend_str, exp_append_str
):
    exp = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy {}".format(exp_append_str),
                "EJ/yr",
                -2.5,
                2.5,
                3.5,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy|Coal {}".format(exp_append_str),
                "EJ/yr",
                -1.25,
                1.25,
                0.25,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario2",
                "World",
                "Primary Energy {}".format(exp_append_str),
                "EJ/yr",
                -2.5,
                2.5,
                -4.5,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario3",
                "World",
                "Primary Energy {}".format(exp_append_str),
                "EJ/yr",
                0.5,
                -0.5,
                4.5,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            dt.datetime(2005, 1, 1),
            dt.datetime(2010, 1, 1),
            dt.datetime(2015, 6, 12),
        ],
    )

    obs = test_processing_scm_df.relative_to_ref_period_mean(
        append_str=tappend_str, **tfilter
    )
    pd.testing.assert_frame_equal(exp.set_index(obs.index.names), obs, check_like=True)


def test_append(test_scm_run):
    test_scm_run.set_meta([5, 6, 7], name="col1")
    other = test_scm_run.filter(scenario="a_scenario2").copy().rename(
        {"variable": {"Primary Energy": "Primary Energy clone"}}
    )

    other.set_meta(2, name="col1")
    other.set_meta("b", name="col2")

    df = test_scm_run.append(other)
    assert isinstance(df, ScmRun)

    # check that the new meta.index is updated, but not the original one
    assert "col1" in test_scm_run.meta_attributes

    # assert that merging of meta works as expected
    npt.assert_array_equal(
        df.meta.sort_values(["scenario", "variable"])["col1"].values, [5, 6, 7, 2]
    )
    pd.testing.assert_series_equal(
        df.meta.sort_values(["scenario", "variable"])["col2"].reset_index(drop=True),
        pd.Series([np.nan, np.nan, np.nan, "b"]),
        check_names=False,
    )

    # assert that appending data works as expected
    ts = df.timeseries().sort_index()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])
    pd.testing.assert_index_equal(
        df.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "col1",
                "col2",
            ]
        ),
    )


def test_append_exact_duplicates(test_scm_run):
    other = copy.deepcopy(test_scm_run)
    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        test_scm_run.append(other).timeseries()

    assert len(mock_warn_taking_average) == 1  # test message elsewhere

    assert_scmdf_almost_equal(test_scm_run, other)


def test_append_duplicates(test_scm_run):
    other = copy.deepcopy(test_scm_run)
    other["time"] = [2020, 2030, 2040]

    res = test_scm_run.append(other)

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 7.0, 2.0, 7.0, 7.0]
    npt.assert_array_equal(
        res["year"], [2005, 2010, 2015, 2020, 2030, 2040]
    )
    npt.assert_almost_equal(obs, exp)


def test_append_duplicates_order_doesnt_matter(test_scm_run):
    other = copy.deepcopy(test_scm_run)
    other["time"] = [2020, 2030, 2040]
    other._ts[2][2] = 5.0

    res = other.append(test_scm_run)

    obs = res.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2.0, 7.0, 7.0, 2.0, 7.0, 5.0]
    npt.assert_array_equal(
        res._time_points.years(), [2005, 2010, 2015, 2020, 2030, 2040]
    )
    npt.assert_almost_equal(obs, exp)


@pytest.mark.parametrize("duplicate_msg", ("warn", "return", False))
def test_append_duplicate_times(test_append_scm_dfs, duplicate_msg):
    base = test_append_scm_dfs["base"]
    other = test_append_scm_dfs["other"]
    expected = test_append_scm_dfs["expected"]

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        res = base.append(other, duplicate_msg=duplicate_msg)

    if duplicate_msg == "warn":
        warn_msg = (
            "Duplicate time points detected, the output will be the average of the "
            "duplicates. Set `duplicate_msg='return'` to examine the "
            "joint timeseries (the duplicates can be found by looking at "
            "`res[res.index.duplicated(keep=False)].sort_index()`. Set "
            "`duplicate_msg=False` to silence this message."
        )
        assert len(mock_warn_taking_average) == 1
        assert str(mock_warn_taking_average[0].message) == warn_msg
    elif duplicate_msg == "return":
        warn_msg = "returning a `pd.DataFrame`, not an `ScmRun`"
        assert len(mock_warn_taking_average) == 1
        assert str(mock_warn_taking_average[0].message) == warn_msg
    else:
        assert not mock_warn_taking_average

    if duplicate_msg == "return":
        # check res gives all timeseries back
        assert res.shape[0] == len(base) + len(other)

        # check advice given in message actually only finds duplicate rows
        look_df = res.meta[res.meta.duplicated(keep=False)]
        assert look_df.shape[0] == 2 * test_append_scm_dfs["duplicate_rows"]
    else:
        pd.testing.assert_frame_equal(
            res.timeseries(), expected.timeseries(), check_like=True
        )


def test_append_doesnt_warn_if_continuous_times(test_append_scm_dfs):
    join_year = 2011
    base = test_append_scm_dfs["base"].filter(year=range(1, join_year))
    other = test_append_scm_dfs["other"].filter(year=range(join_year, 30000))

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        base.append(other)

    assert len(mock_warn_taking_average) == 0


def test_append_doesnt_warn_if_different(test_append_scm_dfs):
    base = test_append_scm_dfs["base"].filter(scenario="a_scenario")
    other = test_append_scm_dfs["base"].filter(scenario="a_scenario2")

    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        base.append(other)

    assert len(mock_warn_taking_average) == 0


def test_append_duplicate_times_error_msg(test_scm_run):
    other = copy.deepcopy(test_scm_run)
    other._data *= 2

    error_msg = re.escape("Unrecognised value for duplicate_msg")
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run.append(other, duplicate_msg="junk")


def test_append_inplace(test_scm_run):
    other = test_scm_run * 2

    obs = test_scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [2, 7, 7]
    npt.assert_almost_equal(obs, exp)
    with warnings.catch_warnings(record=True) as mock_warn_taking_average:
        test_scm_run.append(other, inplace=True)

    assert len(mock_warn_taking_average) == 1  # test message elsewhere

    obs = test_scm_run.filter(scenario="a_scenario2").timeseries().squeeze()
    exp = [(2.0 + 4.0) / 2, (7.0 + 14.0) / 2, (7.0 + 14.0) / 2]
    npt.assert_almost_equal(obs, exp)


def get_append_col_order_time_dfs(base):
    other_2 = base.filter(variable="Primary Energy|Coal").copy()
    base["runmodus"] = "co2_only"
    other = base.copy()

    other._ts[1].meta["variable"] = "Primary Energy|Gas"
    other["time"] = [
        dt.datetime(2002, 1, 1, 0, 0),
        dt.datetime(2008, 1, 1, 0, 0),
        dt.datetime(2009, 1, 1, 0, 0),
    ]

    other_2["ecs"] = 3.0
    other_2["climate_model"] = "a_model2"

    exp = ScmRun(
        pd.DataFrame(
            np.array(
                [
                    [1.0, 1.0, 6.0, 6.0, 6.0, 6.0],
                    [np.nan, 0.5, np.nan, np.nan, 3.0, 3.0],
                    [np.nan, 0.5, np.nan, np.nan, 3.0, 3.0],
                    [0.5, np.nan, 3.0, 3.0, np.nan, np.nan],
                    [2.0, 2.0, 7.0, 7.0, 7.0, 7.0],
                ]
            ).T,
            index=[2002, 2005, 2008, 2009, 2010, 2015],
        ),
        columns={
            "model": ["a_iam"],
            "climate_model": ["a_model", "a_model", "a_model2", "a_model", "a_model"],
            "scenario": [
                "a_scenario",
                "a_scenario",
                "a_scenario",
                "a_scenario",
                "a_scenario2",
            ],
            "region": ["World"],
            "variable": [
                "Primary Energy",
                "Primary Energy|Coal",
                "Primary Energy|Coal",
                "Primary Energy|Gas",
                "Primary Energy",
            ],
            "unit": ["EJ/yr"],
            "runmodus": ["co2_only", "co2_only", np.nan, "co2_only", "co2_only"],
            "ecs": [np.nan, np.nan, 3.0, np.nan, np.nan],
        },
    )

    return base, other, other_2, exp


def test_append_column_order_time_interpolation(test_scm_run):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_run)

    res = df_append([test_scm_run, other, other_2])

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_df_append_inplace_wrong_base(test_scm_run):
    error_msg = "Can only append inplace to an ScmRun"
    with pytest.raises(TypeError, match=error_msg):
        with warnings.catch_warnings(record=True):  # ignore warnings in this test
            df_append([test_scm_run.timeseries(), test_scm_run], inplace=True)


def test_append_chain_column_order_time_interpolation(test_scm_run):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_run)

    res = test_scm_run.append(other).append(other_2)

    pd.testing.assert_frame_equal(
        res.timeseries().sort_index(),
        exp.timeseries().reorder_levels(res.timeseries().index.names).sort_index(),
        check_like=True,
    )


def test_append_inplace_column_order_time_interpolation(test_scm_run):
    base, other, other_2, exp = get_append_col_order_time_dfs(test_scm_run)

    test_scm_run.append(other, inplace=True)
    test_scm_run.append(other_2, inplace=True)

    pd.testing.assert_frame_equal(
        test_scm_run.timeseries().sort_index(),
        exp.timeseries()
        .reorder_levels(test_scm_run.timeseries().index.names)
        .sort_index(),
        check_like=True,
    )


def test_append_inplace_preexisinting_nan(test_scm_run):
    other = copy.deepcopy(test_scm_run)
    other._data *= 2
    other._meta["climate_model"] = "a_model2"
    other.set_meta(np.nan, name="junk")

    original_ts = test_scm_run.timeseries().copy()
    res = test_scm_run.append(other)

    # make sure underlying hasn't changed when not appending inplace
    pd.testing.assert_frame_equal(original_ts, test_scm_run.timeseries())

    exp = pd.concat([test_scm_run.timeseries(), other.timeseries()])
    exp["junk"] = np.nan
    exp.set_index("junk", append=True, inplace=True)

    pd.testing.assert_frame_equal(
        res.timeseries().reorder_levels(exp.index.names).sort_index().reset_index(),
        exp.sort_index().reset_index(),
        check_like=True,
    )


def test_interpolate(combo_df):
    combo, df = combo_df
    target_time_points = combo.target

    res = df.interpolate(
        target_time_points,
        interpolation_type=combo.interpolation_type,
        extrapolation_type=combo.extrapolation_type,
    )

    npt.assert_array_almost_equal(res.values.squeeze(), combo.target_values)


def test_time_mean_year_beginning_of_year(test_scm_df_monthly):
    # should be annual mean centred on January 1st of each year
    res = test_scm_df_monthly.time_mean("AS")

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1992, month=1, day=1).values,
        np.average(np.arange(6)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1996, month=1, day=1).values,
        np.average([42, 43, 44]),
    )

    # automate rest of tests
    def group_annual_mean_beginning_of_year(x):
        if x.month <= 6:
            return x.year
        return x.year + 1

    ts_resampled = (
        test_scm_df_monthly.timeseries()
        .T.groupby(group_annual_mean_beginning_of_year)
        .mean()
        .T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 1, 1))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_year(test_scm_df_monthly):
    # should be annual mean (using all values in that year)
    res = test_scm_df_monthly.time_mean("AC")

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1992, month=7, day=1).values,
        np.average(np.arange(12)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1995, month=7, day=1).values,
        np.average(np.arange(36, 45)),
    )

    # automate rest of tests
    def group_annual_mean(x):
        return x.year

    ts_resampled = (
        test_scm_df_monthly.timeseries().T.groupby(group_annual_mean).mean().T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 7, 1))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_year_end_of_year(test_scm_df_monthly):
    # should be annual mean centred on December 31st of each year
    res = test_scm_df_monthly.time_mean("A")

    # test by hand
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1991, month=12, day=31).values,
        np.average(np.arange(6)),
    )
    npt.assert_allclose(
        res.filter(variable="Radiative Forcing", year=1995, month=12, day=31).values,
        np.average(np.arange(42, 45)),
    )

    # automate rest of tests
    def group_annual_mean_end_of_year(x):
        if x.month >= 7:
            return x.year
        return x.year - 1

    ts_resampled = (
        test_scm_df_monthly.timeseries()
        .T.groupby(group_annual_mean_end_of_year)
        .mean()
        .T
    )
    ts_resampled.columns = ts_resampled.columns.map(lambda x: dt.datetime(x, 12, 31))

    pd.testing.assert_frame_equal(res.timeseries(), ts_resampled, check_like=True)


def test_time_mean_unsupported_style(test_scm_df_monthly):
    error_msg = re.escape("`rule` = `junk` is not supported")
    with pytest.raises(ValueError, match=error_msg):
        test_scm_df_monthly.time_mean("junk")


def test_set_meta_no_name(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, test_scm_run.set_meta, s)


def test_set_meta_as_named_series(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    s.name = "meta_values"
    test_scm_run.set_meta(s)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_run.meta.index)
    exp.name = "meta_values"

    obs = test_scm_run["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_as_unnamed_series(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        codes=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    test_scm_run.set_meta(s, name="meta_values")

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_run.meta.index)
    exp.name = "meta_values"

    obs = test_scm_run["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_non_unique_index_fail(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"], ["a", "b"]],
        codes=[[0, 0], [0, 0], [0, 1]],
        names=["model", "scenario", "region"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_run.set_meta, s)


def test_set_meta_non_existing_index_fail(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_iam", "fail_model"], ["a_scenario", "fail_scenario"]],
        codes=[[0, 1], [0, 1]],
        names=["model", "scenario"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_run.set_meta, s)


def test_set_meta_by_df(test_scm_run):
    df = pd.DataFrame(
        [["a_iam", "a_scenario", "World", 1]],
        columns=["model", "scenario", "region", "col"],
    )

    test_scm_run.set_meta(meta=0.3, name="meta_values", index=df)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_run.meta.index)
    exp.name = "meta_values"

    obs = test_scm_run["meta_values"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_values",
            ]
        ),
    )


def test_set_meta_as_series(test_scm_run):
    # TODO: This is a bit yucky. You can pass in a series which isnt the same length as the meta table without warning
    s = pd.Series([0.3, 0.4])
    test_scm_run.set_meta(s, "meta_series")

    exp = pd.Series(data=[0.3, 0.4, np.nan], index=test_scm_run.meta.index)
    exp.name = "meta_series"

    obs = test_scm_run["meta_series"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_series",
            ]
        ),
    )


def test_set_meta_as_float(test_scm_run):
    test_scm_run.set_meta(3.2, "meta_int")

    exp = pd.Series(data=[3.2, 3.2, 3.2], index=test_scm_run.meta.index, name="meta_int")

    obs = test_scm_run["meta_int"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_int",
            ]
        ),
    )


def test_set_meta_as_str(test_scm_run):
    test_scm_run.set_meta("testing", name="meta_str")

    exp = pd.Series(
        data=["testing", "testing", "testing"],
        index=test_scm_run.meta.index,
        name="meta_str",
    )

    obs = test_scm_run["meta_str"]
    pd.testing.assert_series_equal(obs, exp)
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_str",
            ]
        ),
    )


def test_set_meta_as_str_list(test_scm_run):
    test_scm_run.set_meta(["testing", "testing2", "testing2"], name="category")
    obs = test_scm_run.filter(category="testing")
    assert obs["scenario"].unique() == "a_scenario"


def test_set_meta_as_str_by_index(test_scm_run):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"]],
        codes=[[0], [0]],
        names=["model", "scenario"],
    )

    test_scm_run.set_meta("foo", "meta_str", idx)

    obs = pd.Series(test_scm_run["meta_str"].values)
    pd.testing.assert_series_equal(obs, pd.Series(["foo", "foo", None]))
    pd.testing.assert_index_equal(
        test_scm_run.meta.columns,
        pd.Index(
            [
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                "climate_model",
                "meta_str",
            ]
        ),
    )


def test_set_meta_index_coerce_fail(test_scm_run):
    error_msg = re.escape("index cannot be coerced to pd.MultiIndex")
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run.set_meta("foo", "meta_str", np.array([1, 2]))


def test_filter_by_bool(test_scm_run):
    test_scm_run.set_meta([True, False, False], name="exclude")
    obs = test_scm_run.filter(exclude=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_by_int(test_scm_run):
    test_scm_run.set_meta([1, 2, 3], name="test")
    obs = test_scm_run.filter(test=1)
    assert obs["scenario"].unique() == "a_scenario"


def test_rename_variable(test_scm_run):
    mapping = {
        "variable": {
            "Primary Energy": "Primary Energy|Total",
            "Primary Energy|Coal": "Primary Energy|Fossil",
        }
    }

    obs = test_scm_run.rename(mapping)

    exp = pd.Series(
        ["Primary Energy|Total", "Primary Energy|Fossil", "Primary Energy|Total"]
    )
    pd.testing.assert_series_equal(
        obs["variable"], exp, check_index_type=False, check_names=False
    )


def test_rename_variable_inplace(test_scm_run):
    mapping = {
        "variable": {
            "Primary Energy": "Primary Energy|Total",
            "Primary Energy|Coal": "Primary Energy|Fossil",
        }
    }

    test_scm_run.rename(mapping, inplace=True)

    exp = pd.Series(
        ["Primary Energy|Total", "Primary Energy|Fossil", "Primary Energy|Total"]
    )
    pd.testing.assert_series_equal(
        test_scm_run["variable"], exp, check_index_type=False, check_names=False
    )


def test_rename_index_fail(test_scm_run):
    mapping = {"scenario": {"a_scenario": "a_scenario2"}}
    pytest.raises(ValueError, test_scm_run.rename, mapping)


def test_rename_col_fail(test_scm_run):
    fail_col = "junk"
    error_msg = re.escape("Renaming by {} not supported!".format(fail_col))
    with pytest.raises(ValueError, match=error_msg):
        test_scm_run.rename({fail_col: {"hi": "bye"}})


@pytest.mark.parametrize(
    ("target_unit", "input_units", "filter_kwargs", "expected", "expected_units"),
    [
        ("EJ/yr", "EJ/yr", {}, [1.0, 0.5, 2.0], ["EJ/yr", "EJ/yr", "EJ/yr"]),
        ("PJ/yr", "EJ/yr", {}, [1000.0, 500.0, 2000.0], ["PJ/yr", "PJ/yr", "PJ/yr"]),
        (
            "PJ/yr",
            "EJ/yr",
            {"scenario": "a_scenario2"},
            [1.0, 0.5, 2000.0],
            ["EJ/yr", "EJ/yr", "PJ/yr"],
        ),
        (
            "PJ/yr",
            ["EJ/yr", "TJ/yr", "Gt C / yr"],
            {"variable": "Primary Energy|Coal"},
            [1.0, 0.5 * 1e-3, 2.0],
            ["EJ/yr", "PJ/yr", "Gt C / yr"],
        ),
    ],
)
def test_convert_unit(
    test_scm_run, target_unit, input_units, filter_kwargs, expected, expected_units
):
    test_scm_run["unit"] = input_units
    obs = test_scm_run.convert_unit(target_unit, **filter_kwargs)

    exp_units = pd.Series(expected_units, name="unit")

    pd.testing.assert_series_equal(obs["unit"], exp_units, check_less_precise=True)
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert (test_scm_run["unit"] == input_units).all()


def test_convert_unit_unknown_unit(test_scm_run):
    unknown_unit = "Unknown"
    test_scm_run["unit"] = unknown_unit

    error_msg = re.escape(
        "'{}' is not defined in the unit registry".format(unknown_unit)
    )
    with pytest.raises(UndefinedUnitError, match=error_msg):
        test_scm_run.convert_unit("EJ/yr")


def test_convert_unit_dimensionality(test_scm_run):
    error_msg = "Cannot convert from 'exajoule / a' .* to 'kelvin'"
    with pytest.raises(DimensionalityError, match=error_msg):
        test_scm_run.convert_unit("kelvin")


@pytest.mark.xfail
def test_convert_unit_inplace(test_scm_run):
    units = test_scm_run["unit"].copy()

    ret = test_scm_run.convert_unit("PJ/yr", inplace=True)
    assert ret is None

    assert (test_scm_run["unit"] != units).all()
    npt.assert_array_almost_equal(
        test_scm_run.filter(year=2005).values.squeeze(), [1000.0, 500.0, 2000.0]
    )


def test_convert_unit_context(test_scm_run):
    test_scm_run = test_scm_run.filter(
        variable="Primary Energy"
    )  # Duplicated meta if set all 3 ts to the same variable name
    test_scm_run["unit"] = "kg SF5CF3 / yr"
    test_scm_run["variable"] = "SF5CF3"

    obs = test_scm_run.convert_unit("kg CO2 / yr", context="AR4GWP100")
    factor = 17700
    expected = [1.0 * factor, 2.0 * factor]
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert all(obs["unit_context"] == "AR4GWP100")

    error_msg = "Cannot convert from 'SF5CF3 * kilogram / a' ([SF5CF3] * [mass] / [time]) to 'CO2 * kilogram / a' ([carbon] * [mass] / [time])"
    with pytest.raises(DimensionalityError, match=re.escape(error_msg)):
        test_scm_run.convert_unit("kg CO2 / yr")


def test_convert_existing_unit_context(test_scm_run):
    test_scm_run = test_scm_run.filter(
        variable="Primary Energy"
    )  # Duplicated meta if set all 3 ts to the same variable name
    test_scm_run["unit"] = "kg SF5CF3 / yr"
    test_scm_run["variable"] = "SF5CF3"
    test_scm_run.set_meta("AR4GWP100", "unit_context")

    obs = test_scm_run.convert_unit("kg CO2 / yr", context="AR4GWP100")
    factor = 17700
    expected = [1.0 * factor, 2.0 * factor]
    npt.assert_array_almost_equal(obs.filter(year=2005).values.squeeze(), expected)
    assert all(obs["unit_context"] == "AR4GWP100")

    # TODO: warning if unit_context is different


def test_resample():
    df_dts = [
        dt.datetime(2000, 1, 1),
        dt.datetime(2000, 6, 1),
        dt.datetime(2001, 1, 1),
        dt.datetime(2001, 6, 1),
        dt.datetime(2002, 1, 1),
        dt.datetime(2002, 6, 1),
        dt.datetime(2003, 1, 1),
    ]
    df = ScmDataFrame(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=df_dts,
    )
    res = df.resample("AS")

    obs = res.values.squeeze()
    exp = [1.0, 3.0, 5.0, 7.0]
    npt.assert_almost_equal(obs, exp, decimal=1)


def test_resample_long_datetimes():
    df_dts = [dt.datetime(year, 1, 1) for year in np.arange(1700, 2500 + 1, 100)]
    df = ScmDataFrame(
        np.arange(1700, 2500 + 1, 100),
        columns={
            "scenario": ["a_scenario"],
            "model": ["a_model"],
            "region": ["World"],
            "variable": ["Emissions|BC"],
            "unit": ["Mg /yr"],
        },
        index=df_dts,
    )
    res = df.resample("AS")

    obs = res.values.squeeze()
    exp = np.arange(1700, 2500 + 1)
    npt.assert_almost_equal(obs, exp, decimal=1)


def test_init_no_file():
    fname = "/path/to/nowhere"
    error_msg = re.escape("no data file `{}` found!".format(fname))
    with pytest.raises(OSError, match=error_msg):
        ScmDataFrame(fname)


@pytest.mark.parametrize(
    ("test_file", "test_kwargs"),
    [
        (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "test_data",
                "rcp26_emissions.csv",
            ),
            {},
        ),
        (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "test_data",
                "rcp26_emissions.xls",
            ),
            {},
        ),
        (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "test_data",
                "rcp26_emissions_multi_sheet.xlsx",
            ),
            {"sheet_name": "rcp26_emissions"},
        ),
        (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "test_data",
                "rcp26_emissions_multi_sheet_data.xlsx",
            ),
            {},
        ),
    ],
)
def test_read_from_disk(test_file, test_kwargs):
    loaded = ScmDataFrame(test_file, **test_kwargs)
    assert (
        loaded.filter(variable="Emissions|N2O", year=1767).timeseries().values.squeeze()
        == 0.010116813
    )


@pytest.mark.parametrize("separator", ["|", "__", "/", "~", "_", "-"])
def test_separator_changes(test_scm_run, separator):
    variable = test_scm_run["variable"]
    test_scm_run["variable"] = [v.replace("|", separator) for v in variable]

    test_scm_run.data_hierarchy_separator = separator

    pd.testing.assert_series_equal(
        test_scm_run.filter(level=0)["variable"],
        pd.Series(["Primary Energy", "Primary Energy"], index=[0, 2], name="variable"),
    )

    pd.testing.assert_series_equal(
        test_scm_run.filter(level=1)["variable"],
        pd.Series(
            ["Primary Energy{}Coal".format(separator)], index=[1], name="variable"
        ),
    )


def test_get_meta(test_scm_run):
    assert test_scm_run.get_unique_meta("climate_model") == ["a_model"]
    assert test_scm_run.get_unique_meta("variable") == [
        "Primary Energy",
        "Primary Energy|Coal",
    ]


@pytest.mark.parametrize("no_duplicates", [True, False])
def test_get_meta_no_duplicates(test_scm_run, no_duplicates):
    if no_duplicates:
        assert (
            test_scm_run.get_unique_meta("climate_model", no_duplicates=no_duplicates)
            == "a_model"
        )

        error_msg = re.escape(
            "`variable` column is not unique (found values: {})".format(
                test_scm_run["variable"].unique().tolist()
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            test_scm_run.get_unique_meta("variable", no_duplicates=no_duplicates)
    else:
        assert test_scm_run.get_unique_meta(
            "climate_model", no_duplicates=no_duplicates
        ) == ["a_model"]
        assert test_scm_run.get_unique_meta("variable", no_duplicates=no_duplicates) == [
            "Primary Energy",
            "Primary Energy|Coal",
        ]
