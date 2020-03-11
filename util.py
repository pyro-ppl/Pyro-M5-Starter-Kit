# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import zipfile

import numpy as np
import pandas as pd
import torch


class M5Data:
    """
    A helper class to read M5 source files and create submissions.

    :param str data_path: Path to the folder that contains M5 data files, which is
    either a single `.zip` file or some `.csv` files extracted from that zip file.
    """
    def __init__(self, data_path=None):
        self.data_path = os.path.abspath("data") if data_path is None else data_path
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"There is no folder '{self.data_path}'.")

        acc_path = os.path.join(self.data_path, "m5-forecasting-accuracy.zip")
        unc_path = os.path.join(self.data_path, "m5-forecasting-uncertainty.zip")
        self.acc_zipfile = zipfile.ZipFile(acc_path) if os.path.exists(acc_path) else None
        self.unc_zipfile = zipfile.ZipFile(unc_path) if os.path.exists(unc_path) else None

        self._sales_df = None
        self._calendar_df = None
        self._prices_df = None

    @property
    def num_items(self):
        return self.sales_df.shape[0]

    @property
    def num_days(self):
        return self.sales_df.shape[1] - 5 + 2 * 28

    @property
    def num_items_by_state(self):
        return self.sales_df["state_id"].value_counts().to_dict()

    @property
    def event_types(self):
        return ["Cultural", "National", "Religious", "Sporting"]

    @property
    def sales_df(self):
        if self._sales_df is None:
            self._sales_df = self.read_csv("sales_train_validation.csv", index_col=0)
        return self._sales_df

    @property
    def calendar_df(self):
        if self._calendar_df is None:
            self._calendar_df = self.read_csv("calendar.csv", index_col=0)
        return self._calendar_df

    @property
    def prices_df(self):
        if self._prices_df is None:
            df = self.read_csv("sell_prices.csv")
            df["id"] = df.item_id + "_" + df.store_id + "_validation"
            self._prices_df = df
        return self._prices_df

    def listdir(self):
        files = set(os.listdir(self.data_path))
        if self.acc_zipfile:
            files |= set(self.acc_zipfile.namelist())
        if self.unc_zipfile:
            files |= set(self.unc_zipfile.namelist())
        return files

    def read_csv(self, filename, index_col=None):
        """
        Returns the dataframe from csv file ``filename``.

        :param str filename: name of the file with trailing `.csv`.
        :param int index_col: indicates which column from csv file is considered as index.
        """
        assert filename.endswith(".csv")
        if filename not in self.listdir():
            raise FileNotFoundError(f"Cannot find either '{filename}' "
                                    "or 'm5-forecasting-accuracy.zip' file "
                                    f"in '{self.data_path}'.")

        if self.acc_zipfile and filename in self.acc_zipfile.namelist():
            return pd.read_csv(self.acc_zipfile.open(filename), index_col=index_col)

        if self.unc_zipfile and filename in self.unc_zipfile.namelist():
            return pd.read_csv(self.acc_zipfile.open(filename), index_col=index_col)

        return pd.read_csv(os.path.join(self.data_path, filename), index_col=index_col)

    def get_sales(self):
        """
        Returns `sales` torch.Tensor with shape `num_items x num_train_days`.
        """
        return torch.from_numpy(self.sales_df.iloc[:, 5:].values).type(torch.get_default_dtype())

    def get_prices(self, fillna=0.):
        """
        Returns `prices` torch.Tensor with shape `num_items x num_days`.

        In some days, there are some items not available, so their prices will be NaN.

        :param float fillna: a float value to replace NaN. Defaults to 0.
        """
        df = pd.pivot_table(self.prices_df, values="sell_price", index="id", columns="wm_yr_wk")
        df = df.fillna(fillna).loc[self.sales_df.index]
        x = torch.from_numpy(df.values).type(torch.get_default_dtype())
        x = x.repeat_interleave(7, dim=-1)[:, :-5]  # last week only includes 2 days
        assert x.shape == (self.num_items, self.num_days)
        return x

    def get_snap(self):
        """
        Returns a `3 x num_days` boolean tensor which indicates whether
        SNAP purchases are allowed at a state in a particular day. The order
        of the first dimension indicates the states "CA", "TX", "WI" respectively.

        Usage::

            >>> m5 = M5Data()
            >>> snap = m5.get_snap_tensor()
            >>> assert snap.shape == (3, m5.num_days)
            >>> n = m5.num_items_by_state
            >>> snap = snap.repeat_interleave(torch.tensor([n["CA"], n["TX"], n["WI"]]), dim=0)
            >>> assert snap.shape == (m5.num_items, m5.num_days)
        """
        x = torch.from_numpy(self.calendar_df[["snap_CA", "snap_TX", "snap_WI"]].T.values)
        assert x.shape == (3, self.num_days)
        return x

    def get_event(self, by_types=False):
        """
        Returns a tensor with length `num_days` indicating whether there are
        special events on a particular day.

        There are 4 types of events: "Cultural", "National", "Religious", "Sporting".

        :param bool by_types: if True, returns a `num_days x 4` tensor indicating
            special event by type. Otherwise, only returns a 1D tensor indicating
            whether there is a special event.
        """
        if not by_types:
            return torch.from_numpy(self.calendar_df["event_type_1"].notnull().values)

        types = self.event_types
        event1 = pd.get_dummies(self.calendar_df["event_type_1"])[types].astype(bool)
        event2 = pd.DataFrame(columns=types)
        types2 = ["Cultural", "Religious"]
        event2[types2] = pd.get_dummies(self.calendar_df["event_type_2"])[types2].astype(bool)
        event2.fillna(False, inplace=True)
        return torch.from_numpy(event1.values | event2.values)

    def get_christmas(self):
        """
        Returns a boolean 1D tensor with length `num_days` indicating if that day is
        Chrismas.
        """
        return torch.from_numpy(self.get_calendar_df().index.str.endswith("12-25"))

    def get_aggregated_sales(self, state=True, store=True, cat=True, dept=True, item=True):
        """
        Returns aggregated sales.
        """
        groups = []
        if not state:
            groups.append("state_id")
        if not store:
            groups.append("store_id")
        if not cat:
            groups.append("cat_id")
        if not dept:
            groups.append("dept_id")
        if not item:
            groups.append("item_id")

        if len(groups) > 0:
            x = self.sales_df.groupby(groups).sum().values
        else:
            x = self.sales_df.iloc[:, 5:].sum().values[None, :]
        return torch.from_numpy(x).type(torch.get_default_dtype())

    def get_all_aggregated_sales(self):
        xs = []
        for level in self.aggregation_levels:
            xs.append(self.get_aggregated_sales(**level))
        xs = torch.cat(xs, 0)
        assert xs.shape[0] == 42840
        return xs

    @property
    def aggregation_levels(self):
        """
        Returns the list of all aggregation levels.
        """
        return [
            {"state": True,  "store": True,  "cat": True,  "dept": True,  "item": True},
            {"state": False, "store": True,  "cat": True,  "dept": True,  "item": True},
            {"state": True,  "store": False, "cat": True,  "dept": True,  "item": True},
            {"state": True,  "store": True,  "cat": False, "dept": True,  "item": True},
            {"state": True,  "store": True,  "cat": True,  "dept": False, "item": True},
            {"state": False, "store": True,  "cat": False, "dept": True,  "item": True},
            {"state": False, "store": True,  "cat": True,  "dept": False, "item": True},
            {"state": True,  "store": False, "cat": False, "dept": True,  "item": True},
            {"state": True,  "store": False, "cat": True,  "dept": False, "item": True},
            {"state": True,  "store": True,  "cat": True,  "dept": True,  "item": False},
            {"state": False, "store": True,  "cat": True,  "dept": True,  "item": False},
            {"state": False, "store": False, "cat": False, "dept": False, "item": False},
        ]

    def make_accuracy_submission(self, filename, prediction):
        """
        Makes submission file given prediction result.
        """
        submission_df = self.read_csv("sample_submission.csv", index_col=0)
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (self.num_items, 28)
        submission_df.iloc[:self.num_items, :] = prediction
        submission_df.to_csv(filename)

    def make_uncertainty_submission(self, filename, median, quantile_50,
                                    quantile_67, quantile_95, quantile_99):
        """
        Each median is a dict of {level: value}
        """
        # TODO: arrange uncertainty in correct rows
        # avoid duplicated `sample_submission.csv` filename
        pass


class BatchDataLoader:
    """
    DataLoader class which iterates over the dataset (data_x, data_y) in batch.

    Usage::

        >>> data_loader = BatchDataLoader(data_x, data_y, batch_size=1000)
        >>> for batch_x, batch_y in data_loader:
        ...     # do something with batch_x, batch_y
    """
    def __init__(self, data_x, data_y, batch_size, shuffle=True):
        super().__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert self.data_x.size(0) == self.data_y.size(0)
        assert len(self) > 0

    @property
    def size(self):
        return self.data_x.size(0)

    def __len__(self):
        # XXX: should we remove or include the tailing data (which has len < batch_size)?
        return math.ceil(self.size / self.batch_size)

    def _sample_batch_indices(self):
        if self.shuffle:
            idx = torch.randperm(self.size)
        else:
            idx = torch.arange(self.size)
        return idx, len(self)

    def __iter__(self):
        idx, n_batches = self._sample_batch_indices()
        for i in range(n_batches):
            _slice = idx[i * self.batch_size: (i + 1) * self.batch_size]
            yield self.data_x[_slice], self.data_y[_slice]
