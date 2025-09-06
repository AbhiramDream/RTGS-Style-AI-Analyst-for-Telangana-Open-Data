"""
Simple robust imputer.

Behavior:
- Works without sklearn.
- Numeric -> median
- Boolean -> mode
- Categorical/object -> mode (fallback to empty string)
- Datetime -> mode (fallback to min), preserves dtype
- Drops columns with > drop_threshold missing values (optional)
- Attempts to restore integer dtypes when safe
- Non-destructive (works on a copy)
"""

from typing import Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataImputer:
    def __init__(self, df: pd.DataFrame, drop_threshold: Optional[float] = None):
        """
        df: input dataframe (not modified)
        drop_threshold: float in (0,1); if set, drop columns with missing ratio > threshold
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._orig = df.copy()
        self.drop_threshold = drop_threshold

    def impute(self) -> pd.DataFrame:
        df = self._orig.copy()
        if df.empty:
            return df

        # optional drop very-sparse columns
        if self.drop_threshold is not None:
            miss = df.isna().mean()
            to_drop = miss[miss > self.drop_threshold].index.tolist()
            if to_drop:
                logger.info("Dropping columns due to missing ratio > %s: %s", self.drop_threshold, to_drop)
                df.drop(columns=to_drop, inplace=True)

        orig_dtypes = self._orig.dtypes.to_dict()

        for col in df.columns:
            series = df[col]
            if series.isnull().sum() == 0:
                continue  # nothing to do

            if pd.api.types.is_bool_dtype(series):
                # boolean -> fill with mode (True/False)
                mode = series.mode()
                fill = bool(mode.iloc[0]) if not mode.empty else False
                df[col] = series.fillna(fill)

            elif pd.api.types.is_integer_dtype(orig_dtypes[col]) or pd.api.types.is_integer_dtype(series):
                # integer-like -> use median (keeps values integral), fill with median then attempt cast back
                median = series.median(skipna=True)
                if pd.isna(median):
                    fill = 0
                else:
                    fill = int(median)
                df[col] = series.fillna(fill).astype(float)  # keep float for safety; restore later

            elif pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
                # numeric -> median
                median = series.median(skipna=True)
                fill = 0.0 if pd.isna(median) else float(median)
                df[col] = series.fillna(fill)

            elif pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_datetime64_any_dtype(orig_dtypes[col]):
                # datetime -> mode, fallback to min, else NaT
                mode = series.mode()
                if not mode.empty:
                    fill = mode.iloc[0]
                else:
                    non_null = series.dropna()
                    fill = non_null.min() if not non_null.empty else pd.NaT
                df[col] = series.fillna(fill)

            else:
                # object / category / others -> mode or empty string
                mode = series.mode()
                if not mode.empty:
                    fill = mode.iloc[0]
                else:
                    fill = "" if series.dtype == object else series.dtype.type() if hasattr(series.dtype, "type") else ""
                df[col] = series.fillna(fill)

        # final sweep: any remaining NaNs -> column-wise safe defaults
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    df[col] = df[col].fillna(0)
                elif pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                    df[col] = df[col].fillna(pd.NaT)
                else:
                    df[col] = df[col].fillna("")

        # attempt to restore integer dtypes where original was integer
        for col, dtype in orig_dtypes.items():
            if col not in df.columns:
                continue
            try:
                if pd.api.types.is_integer_dtype(dtype) and pd.api.types.is_float_dtype(df[col].dtype):
                    arr = df[col].to_numpy()
                    frac = np.modf(arr)[0]
                    if np.all(np.isfinite(arr)) and np.all(frac == 0):
                        df[col] = df[col].astype(dtype)
            except Exception:
                # ignore failures to cast back
                logger.debug("Could not restore integer dtype for column %s", col)

        return df


# Example usage:
# imputer = DataImputer(df, drop_threshold=0.9)
#