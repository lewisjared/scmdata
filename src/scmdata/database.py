"""
Database for handling large datasets in a performant, but flexible way

Data is chunked using unique combinations of metadata. This allows for the
database to expand as new data is added without having to change any of the
existing data.

Subsets of data are also able to be read without having to load all the data
and then filter. For example, one could save model results from a number of different
climate models and then load just the ``Surface Temperature`` data for all models.
"""
import glob
import itertools
import os
import os.path
import pathlib
import shutil

import pandas as pd
import tqdm.autonotebook as tqdman

from scmdata import ScmRun, run_append


def ensure_dir_exists(fp):
    """
    Ensure directory exists

    Parameters
    ----------
    fp : str
        Filepath of which to ensure the directory exists
    """
    dir_to_check = os.path.dirname(fp)
    if not os.path.isdir(dir_to_check):
        try:
            os.makedirs(dir_to_check)
        except OSError:  # pragma: no cover
            # Prevent race conditions if multiple threads attempt to create dir at same time
            if not os.path.exists(dir_to_check):
                raise


def _check_is_subdir(root, d):
    root_path = pathlib.Path(root).resolve()
    out_path = pathlib.Path(d).resolve()

    is_subdir = root_path in out_path.parents
    # Sanity check that we never mangle anything outside of the root dir
    if not is_subdir:  # pragma: no cover
        raise AssertionError("{} not in {}".format(d, root))


class ScmDatabase:
    """
    On-disk database handler for outputs from SCMs

    Data is split into groups as specified by :attr:`levels`. This allows for fast
    reading and writing of new subsets of data when a single output file is no longer
    performant or data cannot all fit in memory.
    """

    def __init__(
        self, root_dir, levels=("climate_model", "variable", "region", "scenario"),
    ):
        """
        Initialise the database

        Parameters
        ----------
        root_dir : str
            The root directory of the database

        levels : tuple of str
            Specifies how the runs should be stored on disk.

            The data will be grouped by ``levels``. These levels should be adapted to
            best match the input data and desired access pattern. If there are any
            additional varying dimensions, they will be stored as dimensions.

        .. note::

            Creating a new :class:`ScmDatabase` does not modify any existing data on
            disk. To load an existing database ensure that the :attr:`root_dir` and
            :attr:`levels` are the same as the previous instance.
        """
        self._root_dir = root_dir
        self.levels = tuple(levels)

    def __repr__(self):
        return "<scmdata.database.SCMDatabase (root_dir: {}, levels: {})>".format(
            self._root_dir, self.levels
        )

    @property
    def root_dir(self):
        """
        Root directory of the database.

        Returns
        -------
        str
        """
        return self._root_dir

    @staticmethod
    def _get_disk_filename(inp):
        def safe_char(c):
            if c.isalnum() or c in "-/*_.":
                return c
            else:
                return "-"

        return "".join(safe_char(c) for c in inp)

    def save(self, scmrun, disable_tqdm=False):
        """
        Save data to the database

        The results are saved with one file for each unique combination of
        :attr:`levels` in a directory structure underneath ``root_dir``.

        Use :meth:`available_data` to see what data is available. Subsets of
        data can then be loaded as an :class:`scmdata.ScmRun <scmdata.run.ScmRun>` using :meth:`load`.

        Parameters
        ----------
        scmrun : :class:`scmdata.ScmRun <scmdata.run.ScmRun>`
            Data to save.

            The timeseries in this run should have valid metadata for each
            of the columns specified in ``levels``.
        disable_tqdm: bool
            If True, do not show the progress bar
        """
        for r in tqdman.tqdm(
            scmrun.groupby(self.levels),
            leave=False,
            desc="Saving to database",
            disable=disable_tqdm,
        ):
            self._save_to_database_single_file(r)

    def _get_out_filepath(self, **levels):
        """
        Get filepath in which data has been saved

        The filepath is the root directory joined with the other information provided.
        The filepath is also cleaned to remove spaces and special characters.

        Parameters
        ----------
        levels: dict of str : str
            The unique value for each level in :attr:`levels'

        Returns
        -------
        str
            Path in which to save the data without spaces or special characters.

        Raises
        ------
        ValueError
            If no value is provided for level in :attr:`levels'
        """
        out_levels = []
        for level in self.levels:
            if level not in levels:
                raise ValueError("expected value for level: {}".format(level))
            out_levels.append(str(levels[level]))

        out_path = os.path.join(self._root_dir, *out_levels)
        out_fname = "__".join(out_levels) + ".nc"
        out_fname = os.path.join(out_path, out_fname)

        _check_is_subdir(self._root_dir, out_fname)

        return self._get_disk_filename(out_fname)

    def _save_to_database_single_file(self, scmrun):
        levels = {
            level: scmrun.get_unique_meta(level, no_duplicates=True).replace(
                os.sep, "_"
            )
            for level in self.levels
        }
        out_file = self._get_out_filepath(**levels)

        ensure_dir_exists(out_file)
        if os.path.exists(out_file):
            existing_run = ScmRun.from_nc(out_file)

            scmrun = run_append([existing_run, scmrun])

        # Check for required extra dimensions
        nunique_meta_vals = scmrun.meta.nunique()
        dimensions = nunique_meta_vals[nunique_meta_vals > 1].index.tolist()
        scmrun.to_nc(out_file, dimensions=dimensions)

    def load(self, disable_tqdm=False, **filters):
        """
        Load data from the database

        Parameters
        ----------
        disable_tqdm: bool
            If True, do not show the progress bar
        filters: dict of str : [str, list[str]]
            Filters for the data to load.

            Defaults to loading all values for a level if it isn't specified.

            If a filter is a list then OR logic is applied within the level.
            For example, if we have ``scenario=["ssp119", "ssp126"]`` then
            both the ssp119 and ssp126 scenarios will be loaded.

        Returns
        -------
        :class:`scmdata.ScmRun <scmdata.run.ScmRun>`
            Loaded data

        Raises
        ------
        ValueError
            If a filter for a level not in :attr:`levels` is specified

            If no data matching ``filters`` is found
        """
        for level in filters:
            if level not in self.levels:
                raise ValueError("Unknown level: {}".format(level))

            if "/" in filters[level]:
                filters[level] = filters[level].replace("/", "_")

        level_options = []
        for level in self.levels:
            level_values = filters.get(level, ["*"])
            if isinstance(level_values, str):
                level_values = [level_values]

            level_options.append(level_values)

        # AND logic across levels, OR logic within levels
        level_options_product = itertools.product(*level_options)
        globs_to_check = [
            self._get_disk_filename(os.path.join(self._root_dir, *levels, "*.nc"))
            for levels in level_options_product
        ]

        load_files = [
            v
            for vlist in [glob.glob(g, recursive=True) for g in globs_to_check]
            for v in vlist
        ]

        return run_append(
            [
                ScmRun.from_nc(f)
                for f in tqdman.tqdm(
                    load_files, desc="Loading files", leave=False, disable=disable_tqdm
                )
            ]
        )

    def delete(self, **filters):
        """
        Delete data from the database

        Parameters
        ----------
        filters: dict of str
            Filters for the data to load.

            Defaults to deleting all data if nothing is specified.

        Raises
        ------
        ValueError
            If a filter for a level not in :attr:`levels` is specified
        """
        for level in filters:
            if level not in self.levels:
                raise ValueError("Unknown level: {}".format(level))
            if "/" in filters[level]:
                filters[level] = filters[level].replace("/", "_")

        paths_to_load = [filters.get(level, "*") for level in self.levels]
        load_path = os.path.join(self._root_dir, *paths_to_load)
        glob_to_use = self._get_disk_filename(load_path)
        load_dirs = glob.glob(glob_to_use, recursive=True)

        for d in load_dirs:
            _check_is_subdir(self._root_dir, d)
            shutil.rmtree(d)

    def available_data(self):
        """
        Get all the data which is available to be loaded

        If metadata includes non-alphanumeric characters then it
        might appear modified in the returned table. The original
        metadata values can still be used to filter data.

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        load_path = os.path.join(self._root_dir, "**", "*.nc")
        all_files = glob.glob(load_path, recursive=True)

        file_meta = []
        for f in all_files:
            dirnames = f.split(os.sep)[:-1]
            file_meta.append(dirnames[-len(self.levels) :])

        data = pd.DataFrame(file_meta, columns=self.levels)

        return data.sort_values(by=data.columns.to_list()).reset_index(drop=True)
