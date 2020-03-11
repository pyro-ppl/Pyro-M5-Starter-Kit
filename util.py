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

        sale_df = self.load_file("sales_train_validation.csv", index_col=0)
        self.index = sale_df.index
        self.num_items = sale_df.shape[0]
        self.num_days = sale_df.shape[1] - 5 + 2 * 28
        self.num_items_by_state = sale_df["state_id"].value_counts().to_dict()

    def listdir(self):
        files = set(os.listdir(self.data_path))
        if self.acc_zipfile:
            files |= set(self.acc_zipfile.namelist())
        if self.unc_zipfile:
            files |= set(self.unc_zipfile.namelist())
        return files

    def load_file(self, filename, index_col=None):
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

    def load_calendar(self):
        """
        Returns `calendar` dataframe.
        """
        df = self.load_file("calendar.csv", index_col=0)
        assert df.shape[0] == self.num_days
        return df

    def load_sales_tensor(self):
        """
        Returns `sales` torch.Tensor with shape `num_items x num_train_days`.
        """
        df = self.load_file("sales_train_validation.csv", index_col=0)
        return torch.from_numpy(df.iloc[:, 5:].values).type(torch.get_default_dtype())

    def load_prices_tensor(self, fillna=0.):
        """
        Returns `prices` torch.Tensor with shape `num_items x num_days`.

        In some days, there are some items not available, so their prices will be NaN.

        :param float fillna: a float value to replace NaN. Defaults to 0.
        """
        df = self.load_file("sell_prices.csv")
        df["id"] = df.item_id + "_" + df.store_id + "_validation"
        df = pd.pivot_table(df, values="sell_price", index="id", columns="wm_yr_wk").fillna(fillna)
        df = df.loc[self.index]
        x = torch.from_numpy(df.values).type(torch.get_default_dtype())
        x = x.repeat_interleave(7, dim=-1)[:, :-5]  # last week only includes 2 days
        assert x.shape == (self.num_items, self.num_days)
        return x

    def load_snap_tensor(self):
        """
        Returns a boolean tensor which indicating whether SNAP purchases are allowed.
        """
        df = self.load_calendar()
        x = torch.from_numpy(df[["snap_CA", "snap_TX", "snap_WI"]].T.values)
        d = self.num_items_by_state
        x = x.repeat_interleave(torch.tensor([d["CA"], d["TX"], d["WI"]]), dim=0)
        assert x.shape == (self.num_items, self.num_days)
        return x

    def load_aggregated_tensor(self, state=False, store=False, category=False, department=False):
        """
        Returns aggregated sales.
        """
        # TODO: aggregate and return correct index

    def aggregation_levels(self):
        """
        Returns the list of all aggregation levels.
        """
        return [
            {"state": True, "store": True, "category": True, "department": True},
            {"state": False, "store": True, "category": True, "department": True},
            {"state": True, "store": False, "category": True, "department": True},
            {"state": True, "store": True, "category": False, "department": True},
            {"state": True, "store": True, "category": True, "department": False},
            {"state": False, "store": True, "category": False, "department": True},
            {"state": False, "store": True, "category": True, "department": False},
            {"state": True, "store": False, "category": False, "department": True},
            {"state": True, "store": False, "category": True, "department": False},
            {"state": True, "store": True, "category": False, "department": False},
            {"state": False, "store": True, "category": False, "department": False},
            {"state": False, "store": False, "category": False, "department": False},
        ]

    def make_accuracy_submission(self, filename, prediction):
        """
        Makes submission file given prediction result.
        """
        submission_df = self.load_file("sample_submission.csv", index_col=0)
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape == (submission_df.shape[0] // 2, 28)
        submission_df.iloc[:prediction.shape[0], :] = prediction
        submission_df.to_csv(filename)

    def make_uncertainty_submission(self, filename, median, quantile_50,
                                    quantile_67, quantile_95, quantile_99):
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
