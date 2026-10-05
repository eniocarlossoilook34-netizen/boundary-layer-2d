"""Integral (momentum) methods.

Von Kármán momentum integral equation, steady 2D laminar flow:

    d theta/dx + (2 + H) (theta / U_e) dU_e/dx = tau_w / (rho U_e^2)

* Pohlhausen-type profiles: assume u/U_e = F(y/delta) on a flat plate.
* Thwaites (1949): a one-parameter closure that integrates the equation in
  closed form for an arbitrary U_e(x).

Everything is in the same scaled units as the marching solver (nu = 1,
y ~ L / sqrt(Re_L)).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

Array = np.ndarray


# ----------------------------------------------------------------------
# Pohlhausen-type profiles on a flat plate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AssumedProfile:
    """u/U = F(s), s = y/delta in [0, 1]; F(s>1) = 1."""

    name: str
    F: Callable[[Array], Array]
    dF0: float          # F'(0)
    latex: str

    @property
    def alpha(self) -> float:
        """theta / delta = int_0^1 F (1 - F) ds."""
        return quad(lambda s: self.F(s) * (1 - self.F(s)), 0, 1)[0]

    @property
    def alpha_star(self) -> float:
        """delta* / delta = int_0^1 (1 - F) ds."""
        return quad(lambda s: 1 - self.F(s), 0, 1)[0]

    # Momentum integral with dU_e/dx = 0:  alpha d delta/dx = nu F'(0) / (U delta)
    #   => delta/x = sqrt(2 F'(0) / alpha) / sqrt(Re_x)
    @property
    def delta(self) -> float:
        """delta sqrt(Re_x) / x."""
        return float(np.sqrt(2.0 * self.dF0 / self.alpha))

    @property
    def delta_star(self) -> float:
        return self.alpha_star * self.delta

    @property
    def theta(self) -> float:
        return self.alpha * self.delta

    @property
    def cf(self) -> float:
        """cf sqrt(Re_x) = sqrt(2 alpha F'(0))."""
        return float(np.sqrt(2.0 * self.alpha * self.dF0))

    @property
    def H(self) -> float:
        return self.alpha_star / self.alpha

    def profile(self, s: Array) -> Array:
        s = np.asarray(s, dtype=float)
        return np.where(s < 1.0, self.F(np.clip(s, 0, 1)), 1.0)


PROFILES: tuple[AssumedProfile, ...] = (
    AssumedProfile("Linear", lambda s: s, 1.0, r"$s$"),
    AssumedProfile("Quadrático", lambda s: 2 * s - s**2, 2.0, r"$2s - s^2$"),
    AssumedProfile("Cúbico", lambda s: 1.5 * s - 0.5 * s**3, 1.5,
                   r"$\frac{3}{2}s - \frac{1}{2}s^3$"),
    AssumedProfile("Quártico", lambda s: 2 * s - 2 * s**3 + s**4, 2.0,
                   r"$2s - 2s^3 + s^4$"),
    AssumedProfile("Senoidal", lambda s: np.sin(np.pi * s / 2), np.pi / 2,
                   r"$\sin(\pi s/2)$"),
)


def profile_by_name(name: str) -> AssumedProfile:
    for p in PROFILES:
        if p.name.lower() == name.lower():
            return p
    raise KeyError(name)


# ----------------------------------------------------------------------
# Thwaites' method
# ----------------------------------------------------------------------
LAMBDA_SEPARATION = -0.0898
"""lambda at which the Cebeci–Bradshaw shear correlation l(lambda) vanishes."""


def thwaites_l(lam: Array) -> Array:
    """Shear correlation l(lambda) = tau_w theta / (mu U_e) (Cebeci & Bradshaw, 1977)."""
    lam = np.asarray(lam, dtype=float)
    fav = 0.22 + 1.57 * lam - 1.8 * lam**2
    adv = 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107)
    return np.where(lam >= 0.0, fav, adv)


def thwaites_H(lam: Array) -> Array:
    """Shape-factor correlation H(lambda) (Cebeci & Bradshaw, 1977)."""
    lam = np.asarray(lam, dtype=float)
    fav = 2.61 - 3.75 * lam + 5.24 * lam**2
    adv = 2.088 + 0.0731 / (lam + 0.14)
    return np.where(lam >= 0.0, fav, adv)


@dataclass
class ThwaitesResult:
    x: Array
    ue: Array
    theta: Array
    lam: Array
    x_separation: float | None

    @property
    def H(self) -> Array:
        return thwaites_H(self.lam)

    @property
    def delta_star(self) -> Array:
        return self.H * self.theta

    @property
    def tau_wall(self) -> Array:
        """Scaled wall shear du/dy|_0 = l U_e / theta."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return thwaites_l(self.lam) * self.ue / self.theta

    @property
    def cf(self) -> Array:
        """cf sqrt(Re_L)."""
        return 2.0 * self.tau_wall / self.ue**2


def thwaites(ue: Callable[[Array], Array], x: Array,
             due: Callable[[Array], Array] | None = None,
             theta0_sq: float = 0.0, a: float = 0.45, b: float = 6.0) -> ThwaitesResult:
    """Thwaites' method for an arbitrary outer velocity.

        theta^2 U_e^b = theta_0^2 U_e0^b + a int_{x0}^{x} U_e^{b-1} dx

    ``x`` should start at the leading edge (or stagnation point) and be fine
    enough for the trapezoidal quadrature. The result is truncated at the
    first station past separation (lambda < LAMBDA_SEPARATION).
    """
    x = np.asarray(x, dtype=float)
    if due is None:
        def due(xx, _h=1e-6):
            return (ue(xx + _h) - ue(xx - _h)) / (2 * _h)
    U = np.asarray(ue(x), dtype=float) * np.ones_like(x)
    dU = np.asarray(due(x), dtype=float) * np.ones_like(x)

    integral = cumulative_trapezoid(U ** (b - 1.0), x, initial=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_sq = (theta0_sq * U[0] ** b + a * integral) / U**b
    # Stagnation point (U_e = 0 at x0): analytic limit theta^2 = a / (b U_e')
    if U[0] == 0.0:
        theta_sq[0] = a / (b * dU[0])
    lam = theta_sq * dU

    x_sep = None
    below = np.nonzero(lam < LAMBDA_SEPARATION)[0]
    end = len(x)
    if below.size:
        k = below[0]
        x_sep = float(np.interp(LAMBDA_SEPARATION, [lam[k], lam[k - 1]], [x[k], x[k - 1]]))
        end = k
    sl = slice(0, end)
    return ThwaitesResult(x=x[sl], ue=U[sl], theta=np.sqrt(theta_sq[sl]),
                          lam=lam[sl], x_separation=x_sep)
