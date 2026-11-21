from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.special import ndtri


class QuantileScaler:
    # Per feature: piecewise-linear map from value knots to standard-normal
    # scores, clamped at the outer knots. Each knot's score is the normal
    # quantile of its midrank CDF, (F(x-) + F(x)) / 2, so tied values (a
    # zero-inflated counter's zeros, protocol 6/17) land at the middle of
    # their probability mass. sklearn's QuantileTransformer sends ties at the
    # minimum to the lower bound instead (z ~ -5.2), which turns every
    # binary or zero-heavy feature into a +-5 step and dominates the MSE.
    # The knots are plain arrays so that Mantis can apply the same map.

    def __init__(
        self, n_quantiles: int = 1000, subsample: int = 1_000_000, random_state: int = 0
    ) -> None:
        self.n_quantiles = n_quantiles
        self.subsample = subsample
        self.random_state = random_state
        self.feature_names_in_: Optional[np.ndarray] = None
        self.knots_x_: List[np.ndarray] = []
        self.knots_z_: List[np.ndarray] = []
        self.n_samples_fit_: int = 0

    def fit(self, df: pd.DataFrame) -> "QuantileScaler":
        values = df.to_numpy(dtype=np.float64)
        if len(values) > self.subsample:
            rng = np.random.default_rng(self.random_state)
            values = values[rng.choice(len(values), self.subsample, replace=False)]
        self.feature_names_in_ = np.asarray(df.columns, dtype=object)
        self.n_samples_fit_ = len(values)
        probs = np.linspace(0.0, 1.0, self.n_quantiles)
        self.knots_x_, self.knots_z_ = [], []
        for col in values.T:
            ordered = np.sort(col)
            knots = np.unique(np.quantile(ordered, probs, method="inverted_cdf"))
            below = np.searchsorted(ordered, knots, side="left")
            upto = np.searchsorted(ordered, knots, side="right")
            self.knots_x_.append(knots)
            self.knots_z_.append(ndtri((below + upto) / (2.0 * len(ordered))))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        values = df[list(self.feature_names_in_)].to_numpy(dtype=np.float64)
        out = np.empty_like(values)
        for i, (kx, kz) in enumerate(zip(self.knots_x_, self.knots_z_)):
            out[:, i] = np.interp(values[:, i], kx, kz)
        return out

    def same_as(self, other: "QuantileScaler") -> bool:
        return (
            list(self.feature_names_in_) == list(other.feature_names_in_)
            and len(self.knots_x_) == len(other.knots_x_)
            and all(
                a.shape == b.shape and np.allclose(a, b)
                for a, b in zip(
                    self.knots_x_ + self.knots_z_, other.knots_x_ + other.knots_z_
                )
            )
        )
