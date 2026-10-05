"""Empirical turbulent flat-plate correlations, used for context only.

Unlike the laminar results in this package, none of these follow from the
Prandtl equations alone: they need a closure (here, measured profiles and
friction laws). They are included to place the laminar solution on the
full c_f x Re_x map.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

RE_TRANSITION = 5.0e5
"""Customary critical Reynolds number for a smooth flat plate."""


def power_law_profile(s: Array, n: float = 7.0) -> Array:
    """Prandtl's u/U = (y/delta)^(1/n)."""
    s = np.asarray(s, dtype=float)
    return np.where(s < 1.0, np.clip(s, 0, 1) ** (1.0 / n), 1.0)


def delta_power_law(re_x: Array) -> Array:
    """delta / x = 0.37 Re_x^(-1/5) (1/7 power law + Blasius friction law)."""
    return 0.37 * np.asarray(re_x, dtype=float) ** -0.2


def cf_power_law(re_x: Array) -> Array:
    """Local c_f = 0.0592 Re_x^(-1/5), valid for 5e5 < Re_x < 1e7."""
    return 0.0592 * np.asarray(re_x, dtype=float) ** -0.2


def cf_white(re_x: Array) -> Array:
    """Local c_f = 0.455 / ln^2(0.06 Re_x) (White, 2006)."""
    return 0.455 / np.log(0.06 * np.asarray(re_x, dtype=float)) ** 2


def cf_laminar(re_x: Array) -> Array:
    """Blasius c_f = 0.664 / sqrt(Re_x)."""
    return 0.664 / np.sqrt(np.asarray(re_x, dtype=float))
