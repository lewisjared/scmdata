"""
ScmDataFrame provides a high level analysis tool for simple climate model relevant
data. It provides a simple interface for reading/writing, subsetting and visualising
model data. ScmDataFrames are able to hold multiple model runs which aids in analysis of
ensembles of model runs.
"""
import copy
import datetime as dt
import functools
import numbers
import os
import warnings
from logging import getLogger
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from dateutil import parser
from xarray.core.ops import inject_binary_ops

from .dataframe import ScmDataFrame
from .filters import (
    HIERARCHY_SEPARATOR,
    datetime_match,
    day_match,
    hour_match,
    month_match,
    pattern_match,
    years_match,
)
from .groupby import RunGroupBy
from .offsets import generate_range, to_offset
from .pyam_compat import Axes, IamDataFrame, LongDatetimeIamDataFrame
from .time import TimePoints
from .timeseries import TimeSeries
from .units import UnitConverter

_logger = getLogger(__name__)

REQUIRED_COLS = ["model", "scenario", "region", "variable", "unit"]
"""Minimum metadata columns required by an ScmDataFrame"""


def _read_file(  # pylint: disable=missing-return-doc
    fnames: str, *args: Any, **kwargs: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrame` from a file.

    Parameters
    ----------
    *args
        Passed to :func:`_read_pandas`.
    **kwargs
        Passed to :func:`_read_pandas`.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata
    """
    _logger.info("Reading %s", fnames)

    return _format_data(_read_pandas(fnames, *args, **kwargs))


def _read_pandas(fname: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Read a file and return a :class:`pd.DataFrame`.

    Parameters
    ----------
    fname
        Path from which to read data
    *args
        Passed to :func:`pd.read_csv` if :obj:`fname` ends with '.csv', otherwise passed
        to :func:`pd.read_excel`.
    **kwargs
        Passed to :func:`pd.read_csv` if :obj:`fname` ends with '.csv', otherwise passed
        to :func:`pd.read_excel`.

    Returns
    -------
    :obj:`pd.DataFrame`
        Read data

    Raises
    ------
    OSError
        Path specified by :obj:`fname` does not exist
    """
    if not os.path.exists(fname):
        raise OSError("no data file `{}` found!".format(fname))
    if fname.endswith("csv"):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        xl = pd.ExcelFile(fname)
        if len(xl.sheet_names) > 1 and "sheet_name" not in kwargs:
            kwargs["sheet_name"] = "data"
        df = pd.read_excel(fname, *args, **kwargs)

    return df


# pylint doesn't recognise return statements if they include ','
def _format_data(  # pylint: disable=missing-return-doc
    df: Union[pd.DataFrame, pd.Series]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrame` from :class:`pd.DataFrame` or
    :class:`pd.Series`.

    See docstring of :func:`ScmDataFrame.__init__` for details.

    Parameters
    ----------
    df
        Data to format.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata.

    Raises
    ------
    ValueError
        Not all required metadata columns are present or the time axis cannot be
        understood
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if isinstance(c, str)]
    if str_cols:
        df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    # reset the index if meaningful entries are included there
    if list(df.index.names) != [None]:
        df.reset_index(inplace=True)

    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        missing = list(set(REQUIRED_COLS) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    # check whether data in wide or long format
    if "value" in df.columns:
        df, meta = _format_long_data(df)
    else:
        df, meta = _format_wide_data(df)

    # sort data
    df.sort_index(inplace=True)

    return df, meta


def _format_long_data(df):
    # check if time column is given as `year` (int) or `time` (datetime)
    cols = set(df.columns)
    if "year" in cols and "time" not in cols:
        time_col = "year"
    elif "time" in cols and "year" not in cols:
        time_col = "time"
    else:
        msg = "invalid time format, must have either `year` or `time`!"
        raise ValueError(msg)

    extra_cols = list(set(cols) - set(REQUIRED_COLS + [time_col, "value"]))
    df = df.pivot_table(columns=REQUIRED_COLS + extra_cols, index=time_col).value
    meta = df.columns.to_frame(index=None)
    df.columns = meta.index

    return df, meta


def _format_wide_data(df):
    orig = df.copy()

    cols = set(df.columns) - set(REQUIRED_COLS)
    time_cols, extra_cols = False, []
    for i in cols:
        # if in wide format, check if columns are years (int) or datetime
        if isinstance(i, dt.datetime):
            time_cols = True
        else:
            try:
                float(i)
                time_cols = True
            except (ValueError, TypeError):
                try:
                    try:
                        # most common format
                        dt.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # this is super slow so avoid if possible
                        parser.parse(str(i))  # if no ValueError, this is datetime
                    time_cols = True
                except ValueError:
                    extra_cols.append(i)  # some other string

    if not time_cols:
        msg = "invalid column format, must contain some time (int, float or datetime) columns!"
        raise ValueError(msg)

    df = df.drop(REQUIRED_COLS + extra_cols, axis="columns").T
    df.index.name = "time"
    meta = orig[REQUIRED_COLS + extra_cols].set_index(df.columns)

    return df, meta


def _from_ts(
    df: Any, index: Any = None, **columns: Union[str, bool, float, int, List]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrame` from wide timeseries.

    See docstring of :func:`ScmDataFrame.__init__` for details.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        First dataframe is the data. Second dataframe is metadata

    Raises
    ------
    ValueError
        Not all required columns are present
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if index is not None:
        if isinstance(index, np.ndarray):
            df.index = TimePoints(index).to_index()
        else:
            df.index = index

    # format columns to lower-case and check that all required columns exist
    if not set(REQUIRED_COLS).issubset(columns.keys()):
        missing = list(set(REQUIRED_COLS) - set(columns.keys()))
        raise ValueError("missing required columns `{}`!".format(missing))

    df.index.name = "time"

    num_ts = len(df.columns)
    for c_name, col in columns.items():
        col_list = (
            [col] if isinstance(col, str) or not isinstance(col, Iterable) else col
        )

        if len(col_list) == num_ts:
            continue
        if len(col_list) != 1:
            error_msg = (
                "Length of column '{}' is incorrect. It should be length "
                "1 or {}".format(c_name, num_ts)
            )
            raise ValueError(error_msg)
        columns[c_name] = col_list * num_ts

    meta = pd.DataFrame(columns, index=df.columns)

    return df, meta


class ScmRun:  # pylint: disable=too-many-public-methods
    """
    SCMData's data container for a single run of a SCM.

    """

    data_hierarchy_separator = HIERARCHY_SEPARATOR
    """
    str: String used to define different levels in our data hierarchies.

    By default we follow pyam and use "|". In such a case, emissions of CO2 for energy
    from coal would be "Emissions|CO2|Energy|Coal".
    """

    def __init__(
        self,
        data,
        index: Any = None,
        columns: Optional[Dict[str, list]] = None,
        **kwargs: Any,
    ):
        """
        Initialize.

        Parameters
        ----------
        data: Union[ScmDataFrame, IamDataFrame, pd.DataFrame, pd.Series, np.ndarray, str]
            A pd.DataFrame or data file with IAMC-format data columns, or a numpy array
            of timeseries data if :obj:`columns` is specified. If a string is passed,
            data will be attempted to be read from file.

        index
            Only used if :obj:`columns` is not ``None``. If :obj:`index` is not
            ``None``, too, then this value sets the time index of the
            :class:`ScmDataFrame` instance. If :obj:`index` is ``None`` and
            :obj:`columns` is not ``None``, the index is taken from :obj:`data`.

        columns
            If None, ScmDataFrame will attempt to infer the values from the source.
            Otherwise, use this dict to write the metadata for each timeseries in data.
            For each metadata key (e.g. "model", "scenario"), an array of values (one
            per time series) is expected. Alternatively, providing a list of length 1
            applies the same value to all timeseries in data. For example, if you had
            three timeseries from 'rcp26' for 3 different models 'model', 'model2' and
            'model3', the column dict would look like either 'col_1' or 'col_2':

            .. code:: python

                >>> col_1 = {
                    "scenario": ["rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> col_2 = {
                    "scenario": ["rcp26", "rcp26", "rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> assert pd.testing.assert_frame_equal(
                    ScmDataFrame(d, columns=col_1).meta,
                    ScmDataFrame(d, columns=col_2).meta
                )

        **kwargs:
            Additional parameters passed to :func:`pyam.core._read_file` to read files

        Raises
        ------
        ValueError
            If metadata for ['model', 'scenario', 'region', 'variable', 'unit'] is not
            found. A :class:`ValueError` is also raised if you try to load from multiple
            files at once. If you wish to do this, please use :func:`df_append` instead.

        TypeError
            Timeseries cannot be read from :obj:`data`
        """
        if isinstance(data, ScmRun):
            self._ts = data._ts
            self._time_points = TimePoints(data.time_points)
        else:
            self._init_timeseries(data, index, columns, **kwargs)

    def _init_timeseries(
        self,
        data,
        index: Any = None,
        columns: Optional[Dict[str, list]] = None,
        **kwargs: Any,
    ):
        if columns is not None:
            (_df, _meta) = _from_ts(data, index=index, **columns)
        elif isinstance(data, ScmDataFrame):
            # turn off mypy type checking here as ScmDataFrame isn't defined
            # when mypy does type checking
            (_df, _meta) = (
                data._data,  # pylint: disable=protected-access
                data._meta,  # pylint: disable=protected-access
            )
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            (_df, _meta) = _format_data(data.copy())
        elif (IamDataFrame is not None) and isinstance(data, IamDataFrame):
            (_df, _meta) = _format_data(data.data.copy())
        else:
            if not isinstance(data, str):
                if isinstance(data, list) and isinstance(data[0], str):
                    raise ValueError(
                        "Initialising from multiple files not supported, "
                        "use `scmdata.dataframe.ScmDataFrame.append()`"
                    )
                error_msg = "Cannot load {} from {}".format(type(self), type(data))
                raise TypeError(error_msg)

            (_df, _meta) = _read_file(data, **kwargs)

        self._time_points = TimePoints(_df.index.values)
        _df = _df.astype(float)

        self._ts = []
        time_variable = xr.Variable("time", self._time_points.values)
        for name, attrs in _meta.iterrows():
            ts = TimeSeries(
                data=_df[name], coords=[("time", time_variable)], attrs=attrs
            )
            self._ts.append(ts)

    def copy(self, copy_ts=True):
        """
        Return a :func:`copy.deepcopy` of self.

        Also creates copies the underlying Timeseries data

        Returns
        -------
        :obj:`ScmDataFrame`
            :func:`copy.deepcopy` of ``self``
        """
        ret = copy.copy(self)
        if copy_ts:
            ret._ts = [ts.copy() for ts in self._ts]
        return ret

    def __len__(self) -> int:
        """
        Get the number of timeseries.
        """
        return len(self._ts)

    def __getitem__(self, key: Any) -> Any:
        """
        Get item of self with helpful direct access.

        Provides direct access to "time", "year" as well as the columns in :attr:`meta`.
        If key is anything else, the key will be applied to :attr:`_data`.
        """
        _key_check = (
            [key] if isinstance(key, str) or not isinstance(key, Iterable) else key
        )
        if key == "time":
            return pd.Series(self._time_points.to_index(), dtype="object")
        if key == "year":
            return pd.Series(self._time_points.years())
        if set(_key_check).issubset(self.meta_attributes):
            return self._meta_column(key)

        raise KeyError("I don't know what to do with key: {}".format(key))

    def __setitem__(  # pylint: disable=inconsistent-return-statements
        self, key: Any, value: Any
    ) -> Any:
        """
        Set item of self with helpful direct access.

        Provides direct access to set "time" as well as the columns in :attr:`meta`.
        """
        if key == "time":
            self._time_points = TimePoints(value)
            for ts in self._ts:
                if len(value) != len(ts):
                    raise ValueError(
                        "New time series is the incorrect length (expected: {}, got: {})".format(
                            len(value), len(ts)
                        )
                    )
                ts["time"] = self._time_points.values
            return self._time_points.values
        return self.set_meta(value, name=key)

    def __repr__(self):
        def _indent(s):
            lines = ["\t" + l for l in s.split("\n")]
            return "\n".join(lines)

        meta_str = _indent(self.meta.__repr__())
        time_str = [
            "Start: {}".format(self.time_points[0]),
            "End: {}".format(self.time_points[-1]),
        ]
        time_str = _indent("\n".join(time_str))
        return "<scmdata.ScmRun (timeseries: {}, timepoints: {})>\nTime:\n{}\nMeta:\n{}".format(
            len(self), len(self.time_points), time_str, meta_str
        )

    @staticmethod
    def _binary_op(
        f: Callable[..., Any], reflexive=False, **kwargs,
    ) -> Callable[..., "ScmRun"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, ScmRun):
                return NotImplemented
            if not isinstance(other, numbers.Number) and len(other) != len(self):
                raise ValueError("Incorrect length")

            ret = self.copy(copy_ts=False)
            ret._ts = [
                (f(ts, other) if not reflexive else f(other, ts)) for ts in self._ts
            ]
            return ret

        return func

    @property
    def meta_attributes(self):
        meta = []
        for ts in self._ts:
            meta.extend(ts.meta.keys())
        return sorted(list(set(meta)))

    @property
    def time_points(self) -> np.ndarray:
        """
        Time points of the data
        """
        return self._time_points.values

    def timeseries(
        self, meta: Optional[List[str]] = None, check_duplicated: bool = True
    ) -> pd.DataFrame:
        """
        Return the data in wide format (same as the timeseries method of
        :class:`pyam.IamDataFrame`).

        Parameters
        ----------
        meta
            The list of meta columns that will be included in the output's MultiIndex.
            If None (default), then all metadata will be used.

        check_duplicated
            If True, an exception is raised if any of the timeseries have duplicated metadata

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame with datetimes as columns and timeseries as rows. Metadata is in
            the index.

        Raises
        ------
        ValueError
            If the metadata are not unique between timeseries
        """
        df = pd.DataFrame(np.asarray(self._ts))
        _meta = self.meta if meta is None else self.meta[meta]
        if check_duplicated and _meta.duplicated().any():
            raise ValueError("Duplicated meta values")

        df.columns = self._time_points.to_index()
        df.index = pd.MultiIndex.from_arrays(_meta.values.T, names=_meta.columns)

        return df

    @property
    def shape(self) -> tuple:
        return (len(self._ts), len(self.time_points))

    @property
    def values(self) -> np.ndarray:
        """
        Timeseries values without metadata

        Calls :func:`timeseries`
        """
        return np.asarray(self._ts)

    @property
    def meta(self) -> pd.DataFrame:
        """
        Metadata
        """
        return pd.DataFrame(
            [ts.meta for ts in self._ts], index=[ts.name for ts in self._ts]
        )

    def _meta_column(self, col) -> pd.Series:
        vals = []
        for ts in self._ts:
            try:
                vals.append(ts.meta[col])
            except KeyError:
                vals.append(np.nan)

        return pd.Series(vals, name=col, index=[ts.name for ts in self._ts])

    def filter(
        self,
        keep: bool = True,
        inplace: bool = False,
        has_nan: bool = True,
        **kwargs: Any,
    ):
        """
        Return a filtered ScmDataFrame (i.e., a subset of the data).

        Parameters
        ----------
        keep
            If True, keep all timeseries satisfying the filters, otherwise drop all the
            timeseries satisfying the filters

        inplace
            If True, do operation inplace and return None

        has_nan
            If ``True``, convert all nan values in :obj:`meta_col` to empty string
            before applying filters. This means that "" and "*" will match rows with
            :class:`np.nan`. If ``False``, the conversion is not applied and so a search
            in a string column which contains ;class:`np.nan` will result in a
            :class:`TypeError`.

        **kwargs
            Argument names are keys with which to filter, values are used to do the
            filtering. Filtering can be done on:

            - all metadata columns with strings, "*" can be used as a wildcard in search
              strings

            - 'level': the maximum "depth" of IAM variables (number of hierarchy levels,
              excluding the strings given in the 'variable' argument)

            - 'time': takes a :class:`datetime.datetime` or list of
              :class:`datetime.datetime`'s
              TODO: default to np.datetime64

            - 'year', 'month', 'day', hour': takes an :class:`int` or list of
              :class:`int`'s ('month' and 'day' also accept :class:`str` or list of
              :class:`str`)

            If ``regexp=True`` is included in :obj:`kwargs` then the pseudo-regexp
            syntax in :obj:`pattern_match` is disabled.

        Returns
        -------
        :obj:`ScmDataFrame`
            If not :obj:`inplace`, return a new instance with the filtered data.

        Raises
        ------
        AssertionError
            Data and meta become unaligned
        """
        _keep_times, _keep_cols = self._apply_filters(kwargs, has_nan)
        ret = copy.copy(self) if not inplace else self

        if not keep and sum(~_keep_cols) and sum(~_keep_times):
            raise ValueError(
                "If keep=False, filtering cannot be performed on temporally and with "
                "metadata at the same time"
            )

        reduce_times = (~_keep_times).sum() > 0
        reduce_cols = (~_keep_cols).sum() > 0

        if not keep:
            _keep_times = ~_keep_times
            _keep_cols = ~_keep_cols

            if not reduce_cols and not reduce_times:
                # When nothing is filtered, drop everything
                reduce_cols = True

        # Filter the timeseries first
        # I wish lists had the same indexing interface as ndarrays
        if reduce_cols:
            ret._ts = [ret._ts[i] for i, v in enumerate(_keep_cols) if v]

        # Then filter the timeseries if needed
        if reduce_times:
            ret._ts = [ts[_keep_times] for ts in ret._ts]
            ret["time"] = self.time_points[_keep_times]

        if len(ret) == 0:
            _logger.warning("Filtered ScmDataFrame is empty!")

        if not inplace:
            return ret

        return None

    # pylint doesn't recognise ',' in returns type definition
    def _apply_filters(  # pylint: disable=missing-return-doc
        self, filters: Dict, has_nan: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine rows to keep in data for given set of filters.

        Parameters
        ----------
        filters
            Dictionary of filters ``({col: values}})``; uses a pseudo-regexp syntax by
            default but if ``filters["regexp"]`` is ``True``, regexp is used directly.

        has_nan
            If `True``, convert all nan values in :obj:`meta_col` to empty string before
            applying filters. This means that "" and "*" will match rows with
            :class:`np.nan`. If ``False``, the conversion is not applied and so a search
            in a string column which contains :class:`np.nan` will result in a
            :class:`TypeError`.

        Returns
        -------
        :obj:`np.ndarray` of :class:`bool`, :obj:`np.ndarray` of :class:`bool`
            Two boolean :class:`np.ndarray`'s. The first contains the columns to keep
            (i.e. which time points to keep). The second contains the rows to keep (i.e.
            which metadata matched the filters).

        Raises
        ------
        ValueError
            Filtering cannot be performed on requested column
        """
        regexp = filters.pop("regexp", False)
        keep_ts = np.array([True] * len(self.time_points))
        keep_meta = np.array([True] * len(self))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == "variable":
                level = filters["level"] if "level" in filters else None
                keep_meta &= pattern_match(
                    self._meta_column(col),
                    values,
                    level,
                    regexp,
                    has_nan=has_nan,
                    separator=self.data_hierarchy_separator,
                ).values
            elif col in self.meta_attributes:
                keep_meta &= pattern_match(
                    self._meta_column(col),
                    values,
                    regexp=regexp,
                    has_nan=has_nan,
                    separator=self.data_hierarchy_separator,
                ).values
            elif col == "year":
                keep_ts &= years_match(self._time_points.years(), values)

            elif col == "month":
                keep_ts &= month_match(self._time_points.months(), values)

            elif col == "day":
                keep_ts &= self._day_match(values)

            elif col == "hour":
                keep_ts &= hour_match(self._time_points.hours(), values)

            elif col == "time":
                keep_ts &= datetime_match(self._time_points.values, values)

            elif col == "level":
                if "variable" not in filters.keys():
                    keep_meta &= pattern_match(
                        self._meta_column("variable"),
                        "*",
                        values,
                        regexp=regexp,
                        has_nan=has_nan,
                        separator=self.data_hierarchy_separator,
                    ).values
                # else do nothing as level handled in variable filtering

            else:
                raise ValueError("filter by `{}` not supported".format(col))

        return keep_ts, keep_meta

    def _day_match(self, values):
        if isinstance(values, str):
            wday = True
        elif isinstance(values, list) and isinstance(values[0], str):
            wday = True
        else:
            wday = False

        if wday:
            days = self._time_points.weekdays()
        else:  # ints or list of ints
            days = self._time_points.days()

        return day_match(days, values)

    def head(self, *args, **kwargs):
        """
        Return head of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().head()`

        **kwargs
            Passed to :func:`self.timeseries().head()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return tail of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().tail()`

        **kwargs
            Passed to :func:`self.timeseries().tail()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().tail(*args, **kwargs)

    def rename(self, mapping: Dict[str, Dict[str, str]], inplace: bool = False):
        """
        Rename and aggregate column entries using :func:`groupby.sum()` on values. When
        renaming models or scenarios, the uniqueness of the index must be maintained,
        and the function will raise an error otherwise.

        Parameters
        ----------
        mapping
            For each column where entries should be renamed, provide current
            name and target name

            .. code:: python

                {<column name>: {<current_name_1>: <target_name_1>,
                                 <current_name_2>: <target_name_2>}}

        inplace
            If ``True``, do operation inplace and return ``None``

        Returns
        -------
        :obj:`ScmDataFrame`
            If :obj:`inplace` is ``True``, return a new :class:`ScmDataFrame`
            instance

        Raises
        ------
        ValueError
            Column is not in meta or renaming will cause non-unique metadata
        """
        ret = self.copy() if not inplace else self
        for col, _mapping in mapping.items():
            if col not in ret.meta_attributes:
                raise ValueError("Renaming by {} not supported!".format(col))
            for ts in ret._ts:
                orig = ts.meta[col]
                if orig in _mapping:
                    ts.meta[col] = _mapping[orig]

            if ret.meta.duplicated().any():  # pylint: disable=protected-access
                raise ValueError("Renaming to non-unique metadata for {}!".format(col))

        if not inplace:
            return ret

        return None

    def set_meta(
        self,
        meta: Union[pd.Series, list, int, float, str],
        name: Optional[str] = None,
        index: Optional[Union[pd.DataFrame, pd.Series, pd.Index, pd.MultiIndex]] = None,
    ) -> None:
        """
        Set metadata information.

        TODO: re-write this to make it more sane and add type annotations

        Parameters
        ----------
        meta
            Column to be added to metadata

        name
            Meta column name (defaults to :obj:`meta.name`)

        index
            The index to which the metadata is to be applied

        Raises
        ------
        ValueError
            No name can be determined from inputs or index cannot be coerced to
            :class:`pd.MultiIndex`
        """
        # check that name is valid and doesn't conflict with data columns
        # mypy doesn't recognise if
        if (name or (hasattr(meta, "name") and meta.name)) in [  # type: ignore
            None,
            False,
        ]:
            raise ValueError("Must pass a name or use a named pd.Series")
        # mpy doesn't recognise if checks
        name = name or meta.name  # type: ignore

        if index is not None:
            raise NotImplementedError(
                "Removed `index` functionality. Filter and then set meta instead"
            )

        meta = np.atleast_1d(meta)
        if len(meta) == 1:
            for ts in self._ts:
                ts.meta[name] = meta[0]
        elif len(meta) == len(self):
            for i, ts in enumerate(self._ts):
                ts.meta[name] = meta[i]
        else:
            raise ValueError("Invalid shape for metadata")

    def get_unique_meta(
        self, meta: str, no_duplicates: Optional[bool] = False,
    ) -> Union[List[Any], Any]:
        """
        Get unique values in a metadata column.

        Parameters
        ----------
        meta
            Column to retrieve metadata for

        no_duplicates
            Should I raise an error if there is more than one unique value in the
            metadata column?

        Raises
        ------
        ValueError
            There is more than one unique value in the metadata column and
            ``no_duplicates`` is ``True``.

        Returns
        -------
        [List[Any], Any]
            List of unique metadata values. If ``no_duplicates`` is ``True`` the
            metadata value will be returned (rather than a list).
        """
        vals = self[meta].unique().tolist()
        if no_duplicates:
            if len(vals) != 1:
                raise ValueError(
                    "`{}` column is not unique (found values: {})".format(meta, vals)
                )

            return vals[0]

        return vals

    def interpolate(
        self,
        target_times: Union[np.ndarray, List[Union[dt.datetime, int]]],
        interpolation_type: str = "linear",
        extrapolation_type: str = "linear",
    ):
        """
        Interpolate the dataframe onto a new time frame.

        Parameters
        ----------
        target_times
            Time grid onto which to interpolate
        interpolation_type: str
            Interpolation type. Options are 'linear'
        extrapolation_type: str or None
            Extrapolation type. Options are None, 'linear' or 'constant'

        Returns
        -------
        :obj:`ScmDataFrame`
            A new :class:`ScmDataFrame` containing the data interpolated onto the
            :obj:`target_times` grid
        """
        # pylint: disable=protected-access

        target_times = np.asarray(target_times, dtype="datetime64[s]")

        res = self.copy(copy_ts=False)

        res._ts = [
            ts.interpolate(
                target_times,
                interpolation_type=interpolation_type,
                extrapolation_type=extrapolation_type,
            )
            for ts in res._ts
        ]
        res._time_points = TimePoints(target_times)

        return res

    def resample(self, rule: str = "AS", **kwargs: Any):
        """
        Resample the time index of the timeseries data onto a custom grid.

        This helper function allows for values to be easily interpolated onto annual or
        monthly timesteps using the rules='AS' or 'MS' respectively. Internally, the
        interpolate function performs the regridding.

        Parameters
        ----------
        rule
            See the pandas `user guide
            <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
            for a list of options. Note that Business-related offsets such as
            "BusinessDay" are not supported.

        **kwargs
            Other arguments to pass through to :func:`interpolate`

        Returns
        -------
        :obj:`ScmDataFrame`
            New :class:`ScmDataFrame` instance on a new time index

        Examples
        --------
        Resample a dataframe to annual values

        >>> scm_df = ScmDataFrame(
        ...     pd.Series([1, 2, 10], index=(2000, 2001, 2009)),
        ...     columns={
        ...         "model": ["a_iam"],
        ...         "scenario": ["a_scenario"],
        ...         "region": ["World"],
        ...         "variable": ["Primary Energy"],
        ...         "unit": ["EJ/y"],
        ...     }
        ... )
        >>> scm_df.timeseries().T
        model             a_iam
        scenario     a_scenario
        region            World
        variable Primary Energy
        unit               EJ/y
        year
        2000                  1
        2010                 10

        An annual timeseries can be the created by interpolating to the start of years
        using the rule 'AS'.

        >>> res = scm_df.resample('AS')
        >>> res.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2001-01-01 00:00:00       2.001825
        2002-01-01 00:00:00       3.000912
        2003-01-01 00:00:00       4.000000
        2004-01-01 00:00:00       4.999088
        2005-01-01 00:00:00       6.000912
        2006-01-01 00:00:00       7.000000
        2007-01-01 00:00:00       7.999088
        2008-01-01 00:00:00       8.998175
        2009-01-01 00:00:00      10.00000

        >>> m_df = scm_df.resample('MS')
        >>> m_df.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2000-02-01 00:00:00       1.084854
        2000-03-01 00:00:00       1.164234
        2000-04-01 00:00:00       1.249088
        2000-05-01 00:00:00       1.331204
        2000-06-01 00:00:00       1.416058
        2000-07-01 00:00:00       1.498175
        2000-08-01 00:00:00       1.583029
        2000-09-01 00:00:00       1.667883
                                    ...
        2008-05-01 00:00:00       9.329380
        2008-06-01 00:00:00       9.414234
        2008-07-01 00:00:00       9.496350
        2008-08-01 00:00:00       9.581204
        2008-09-01 00:00:00       9.666058
        2008-10-01 00:00:00       9.748175
        2008-11-01 00:00:00       9.833029
        2008-12-01 00:00:00       9.915146
        2009-01-01 00:00:00      10.000000
        [109 rows x 1 columns]


        Note that the values do not fall exactly on integer values as not all years are
        exactly the same length.

        References
        ----------
        See the pandas documentation for
        `resample <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html>`
        for more information about possible arguments.
        """
        orig_dts = self["time"]
        target_dts = generate_range(
            orig_dts.iloc[0], orig_dts.iloc[-1], to_offset(rule)
        )
        return self.interpolate(list(target_dts), **kwargs)

    def time_mean(self, rule: str):
        """
        Take time mean of self

        Note that this method will not copy the ``metadata`` attribute to the returned
        value.

        Parameters
        ----------
        rule : ["AC", "AS", "A"]
            How to take the time mean. The names reflect the pandas
            `user guide <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
            where they can, but only the options
            given above are supported. For clarity, if ``rule`` is ``'AC'``, then the
            mean is an annual mean i.e. each time point in the result is the mean of
            all values for that particular year. If ``rule`` is ``'AS'``, then the
            mean is an annual mean centred on the beginning of the year i.e. each time
            point in the result is the mean of all values from July 1st in the
            previous year to June 30 in the given year. If ``rule`` is ``'A'``, then
            the mean is an annual mean centred on the end of the year i.e. each time
            point in the result is the mean of all values from July 1st of the given
            year to June 30 in the next year.

        Returns
        -------
        :obj:`ScmDataFrame`
            The time mean of ``self``.
        """
        if rule == "AS":

            def group_annual_mean_beginning_of_year(x):
                if x.month <= 6:
                    return x.year
                return x.year + 1

            ts_resampled = (
                self.timeseries()
                .T.groupby(group_annual_mean_beginning_of_year)
                .mean()
                .T
            )
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 1, 1)
            )
            return ScmDataFrame(ts_resampled)

        if rule == "AC":

            def group_annual_mean(x):
                return x.year

            ts_resampled = self.timeseries().T.groupby(group_annual_mean).mean().T
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 7, 1)
            )
            return ScmDataFrame(ts_resampled)

        if rule == "A":

            def group_annual_mean_end_of_year(x):
                if x.month >= 7:
                    return x.year
                return x.year - 1

            ts_resampled = (
                self.timeseries().T.groupby(group_annual_mean_end_of_year).mean().T
            )
            ts_resampled.columns = ts_resampled.columns.map(
                lambda x: dt.datetime(x, 12, 31)
            )
            return ScmDataFrame(ts_resampled)

        raise ValueError("`rule` = `{}` is not supported".format(rule))

    def process_over(
        self, cols: Union[str, List[str]], operation: str, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Process the data over the input columns.

        Parameters
        ----------
        cols
            Columns to perform the operation on. The timeseries will be grouped by all
            other columns in :attr:`meta`.

        operation : ['median', 'mean', 'quantile']
            The operation to perform. This uses the equivalent pandas function. Note
            that quantile means the value of the data at a given point in the cumulative
            distribution of values at each point in the timeseries, for each timeseries
            once the groupby is applied. As a result, using ``q=0.5`` is is the same as
            taking the median and not the same as taking the mean/average.

        **kwargs
            Keyword arguments to pass to the pandas operation

        Returns
        -------
        :obj:`pd.DataFrame`
            The quantiles of the timeseries, grouped by all columns in :attr:`meta`
            other than :obj:`cols`

        Raises
        ------
        ValueError
            If the operation is not one of ['median', 'mean', 'quantile']
        """
        cols = [cols] if isinstance(cols, str) else cols
        ts = self.timeseries()
        group_cols = list(set(ts.index.names) - set(cols))
        grouper = ts.groupby(group_cols)

        if operation == "median":
            return grouper.median(**kwargs)
        if operation == "mean":
            return grouper.mean(**kwargs)
        if operation == "quantile":
            return grouper.quantile(**kwargs)

        raise ValueError("operation must be one of ['median', 'mean', 'quantile']")

    def groupby(self, *group):
        if len(group) == 1 and not isinstance(group[0], str):
            group = tuple(group[0])
        return RunGroupBy(self, group)

    def convert_unit(self, unit: str, context: Optional[str] = None, **kwargs: Any):
        """
        Convert the units of a selection of timeseries.

        Uses :class:`scmdata.units.UnitConverter` to perform the conversion.

        Parameters
        ----------
        unit
            Unit to convert to. This must be recognised by
            :class:`~openscm.units.UnitConverter`.

        context
            Context to use for the conversion i.e. which metric to apply when performing
            CO2-equivalent calculations. If ``None``, no metric will be applied and
            CO2-equivalent calculations will raise :class:`DimensionalityError`.

        **kwargs
            Extra arguments which are passed to :func:`~ScmDataFrame.filter` to
            limit the timeseries which are attempted to be converted. Defaults to
            selecting the entire ScmDataFrame, which will likely fail.

        Returns
        -------
        :obj:`ScmDataFrame`
            If :obj:`inplace` is not ``False``, a new :class:`ScmDataFrame` instance
            with the converted units.
        """
        # pylint: disable=protected-access
        ret = self.copy()

        if "unit_context" not in ret.meta_attributes:
            ret["unit_context"] = None

        to_convert = ret.filter(**kwargs)
        to_not_convert = ret.filter(**kwargs, keep=False)

        def apply_units(group):
            orig_unit = group.get_unique_meta("unit", no_duplicates=True)
            uc = UnitConverter(orig_unit, unit, context=context)
            for ts in group._ts:  # todo: fix when we have an apply function
                ts._data[:] = uc.convert_from(ts._data.values)
            group["unit"] = unit
            group["unit_context"] = context
            return group

        ret = to_convert.groupby("unit").map(apply_units)

        return df_append([ret, to_not_convert])

    def relative_to_ref_period_mean(
        self, append_str: Optional[str] = None, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Return the timeseries relative to a given reference period mean.

        The reference period mean is subtracted from all values in the input timeseries.

        Parameters
        ----------
        append_str
            String to append to the name of all the variables in the resulting DataFrame
            to indicate that they are relevant to a given reference period. E.g. `'rel.
            to 1961-1990'`. If None, this will be autofilled with the keys and ranges of
            :obj:`kwargs`.

        **kwargs
            Arguments to pass to :func:`filter` to determine the data to be included in
            the reference time period. See the docs of :func:`filter` for valid options.

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame containing the timeseries, adjusted to the reference period mean
        """
        ts = self.timeseries()
        # mypy confused by `inplace` default
        ref_period_mean = (
            self.filter(**kwargs).timeseries().mean(axis="columns")  # type: ignore
        )

        res = ts.sub(ref_period_mean, axis="rows")
        res.reset_index(inplace=True)

        if append_str is None:
            append_str = ";".join(
                ["{}: {} - {}".format(k, v[0], v[-1]) for k, v in kwargs.items()]
            )
            append_str = "(ref. period {})".format(append_str)

        res["variable"] = res["variable"].apply(lambda x: "{} {}".format(x, append_str))

        return res.set_index(ts.index.names)

    def append(
        self,
        other,
        inplace: bool = False,
        duplicate_msg: Union[str, bool] = "warn",
        **kwargs: Any,
    ):
        """
        Append additional data to the current dataframe.

        For details, see :func:`df_append`.

        Parameters
        ----------
        other
            Data (in format which can be cast to :class:`ScmDataFrame`) to append

        inplace
            If ``True``, append data in place and return ``None``. Otherwise, return a
            new :class:`ScmDataFrame` instance with the appended data.

        duplicate_msg
            If "warn", raise a warning if duplicate data is detected. If "return",
            return the joint dataframe (including duplicate timeseries) so the user can
            inspect further. If ``False``, take the average and do not raise a warning.

        **kwargs
            Keywords to pass to :func:`ScmDataFrame.__init__` when reading
            :obj:`other`

        Returns
        -------
        :obj:`ScmDataFrame`
            If not :obj:`inplace`, return a new :class:`ScmDataFrame` instance
            containing the result of the append.
        """
        if not isinstance(other, ScmRun):
            other = self.__class__(other, **kwargs)

        return df_append([self, other], inplace=inplace, duplicate_msg=duplicate_msg)

    def to_iamdataframe(self) -> LongDatetimeIamDataFrame:
        """
        Convert to a :class:`LongDatetimeIamDataFrame` instance.

        :class:`LongDatetimeIamDataFrame` is a subclass of :class:`pyam.IamDataFrame`.
        We use :class:`LongDatetimeIamDataFrame` to ensure all times can be handled, see
        docstring of :class:`LongDatetimeIamDataFrame` for details.

        Returns
        -------
        :class:`LongDatetimeIamDataFrame`
            :class:`LongDatetimeIamDataFrame` instance containing the same data.

        Raises
        ------
        ImportError
            If `pyam <https://github.com/IAMconsortium/pyam>`_ is not installed
        """
        if LongDatetimeIamDataFrame is None:
            raise ImportError(
                "pyam is not installed. Features involving IamDataFrame are unavailable"
            )

        return LongDatetimeIamDataFrame(self.timeseries())

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """
        Write timeseries data to a csv file

        Parameters
        ----------
        path
            Path to write the file into
        """
        self.to_iamdataframe().to_csv(path, **kwargs)

    def line_plot(self, x: str = "time", y: str = "value", **kwargs: Any) -> Axes:
        """
        Plot a line chart.

        See :func:`pyam.IamDataFrame.line_plot` for more information.
        """
        return self.to_iamdataframe().line_plot(x, y, **kwargs)  # pragma: no cover

    def scatter(self, x: str, y: str, **kwargs: Any) -> Axes:
        """
        Plot a scatter chart using metadata columns.

        See :func:`pyam.plotting.scatter` for details.
        """
        self.to_iamdataframe().scatter(x, y, **kwargs)  # pragma: no cover

    def region_plot(self, **kwargs: Any) -> Axes:
        """
        Plot regional data for a single model, scenario, variable, and year.

        See :class:`pyam.plotting.region_plot` for details.
        """
        return self.to_iamdataframe().region_plot(**kwargs)  # pragma: no cover

    def pivot_table(
        self,
        index: Union[str, List[str]],
        columns: Union[str, List[str]],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Pivot the underlying data series.

        See :func:`pyam.core.IamDataFrame.pivot_table` for details.
        """
        return self.to_iamdataframe().pivot_table(
            index, columns, **kwargs
        )  # pragma: no cover

    def reduce(self, func, dim=None, axis=None, keep_attrs=None, **kwargs):
        if dim is not None:
            raise ValueError("ScmRun.reduce does not handle dim. Use axis instead")

        input_data = self.values

        if axis is None or axis == 1:
            raise NotImplementedError(
                "Cannot currently reduce along the time dimension"
            )

        if axis is not None:
            data = func(input_data, axis=axis, **kwargs)
        else:
            data = func(input_data, **kwargs)

        if getattr(data, "shape", ()) == self.shape:
            return ScmRun(
                data, index=self.time_points, columns=self.meta.to_dict("list")
            )
        else:
            removed_axes = range(2) if axis is None else np.atleast_1d(axis) % 2
            index = self.time_points
            meta = self.meta.to_dict("list")
            if 0 in removed_axes and len(meta):
                # Reduced the timeseries
                m = self.meta
                n_unique = m.nunique(axis=0)
                m = m.drop(n_unique[n_unique > 1]).drop_duplicates()
                assert len(m) == 1

                meta = m.to_dict("list")

            if 1 in removed_axes:
                raise NotImplementedError  # pragma: no cover

            return ScmRun(data, index=index, columns=meta)


def df_append(
    runs, inplace: bool = False, duplicate_msg: Union[str, bool] = "warn",
):
    """
    Append together many objects.

    When appending many objects, it may be more efficient to call this routine once with
    a list of ScmDataFrames, than using :func:`ScmDataFrame.append` multiple times. If
    timeseries with duplicate metadata are found, the timeseries are appended and values
    falling on the same timestep are averaged (this behaviour can be adjusted with the
    :obj:`duplicate_msg` arguments).

    Parameters
    ----------
    runs:
        The dataframes to append. Values will be attempted to be cast to
        :class:`ScmDataFrame`.

    inplace
        If ``True``, then the operation updates the first item in :obj:`runs` and returns
        ``None``.

    duplicate_msg
        If "warn", raise a warning if duplicate data is detected. If "return", return
        the joint dataframe (including duplicate timeseries) so the user can inspect
        further. If ``False``, take the average and do not raise a warning.

    Returns
    -------
    :obj:`ScmDataFrame`
        If not :obj:`inplace`, the return value is the object containing the merged
        data. The resultant class will be determined by the type of the first object. If
        ``duplicate_msg == "return"``, a `pd.DataFrame` will be returned instead.

    Raises
    ------
    TypeError
        If :obj:`inplace` is ``True`` but the first element in :obj:`dfs` is not an
        instance of :class:`ScmDataFrame`

    ValueError
        :obj:`duplicate_msg` option is not recognised.
    """
    if inplace:
        if not isinstance(runs[0], ScmRun):
            raise TypeError("Can only append inplace to an ScmRun")
        ret = runs[0]
    else:
        ret = runs[0].copy()

    for run in runs[1:]:
        ret._ts.extend(run._ts)

    # Determine the new common timebase
    new_t = np.concatenate([r.time_points for r in runs])
    new_t = np.unique(new_t)
    new_t.sort()

    # reindex if the timebase isn't the same
    all_valid_times = True
    for r in runs:
        if not np.array_equal(new_t, r.time_points):
            all_valid_times = False
    if not all_valid_times:
        ret._ts = [ts.reindex(new_t) for ts in ret._ts]
        ret._time_points = TimePoints(new_t)

    if ret.meta.duplicated().any():
        if duplicate_msg:
            warn_handle_res = _handle_potential_duplicates_in_append(ret, duplicate_msg)
            if warn_handle_res is not None:
                return warn_handle_res  # type: ignore  # special case

        # average identical metadata
        ret._ts = ret.groupby(ret.meta_attributes).mean(axis=0)._ts

    ret._ts.sort(key=lambda a: a.name)

    if not inplace:
        return ret


def _handle_potential_duplicates_in_append(data, duplicate_msg):
    # If only one number contributes to each of the timeseries, we're not looking at
    # duplicates so can return.
    ts = data.timeseries(check_duplicated=False)
    contributing_values = (~ts.isnull()).astype(int).groupby(ts.index.names).sum()
    duplicates = (contributing_values > 1).any().any()
    if not duplicates:
        return None

    if duplicate_msg == "warn":
        warn_msg = (
            "Duplicate time points detected, the output will be the average of "
            "the duplicates. Set `duplicate_msg='return'` to examine the joint "
            "timeseries (the duplicates can be found by looking at "
            "`res[res.index.duplicated(keep=False)].sort_index()`. Set "
            "`duplicate_msg=False` to silence this message."
        )
        warnings.warn(warn_msg)
        return None

    if duplicate_msg == "return":
        warnings.warn("returning a `pd.DataFrame`, not an `ScmRun`")
        return data

    raise ValueError("Unrecognised value for duplicate_msg")


inject_binary_ops(ScmRun)