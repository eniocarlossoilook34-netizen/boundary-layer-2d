"""Similarity solutions of the laminar boundary-layer equations.

The Falkner–Skan family is written in the form

    f''' + f f'' + beta (1 - f'^2) = 0,
    f(0) = f'(0) = 0,   f'(eta -> inf) = 1,

with the similarity variable and stream function

    eta = y * sqrt((m + 1) U_e / (2 nu x)),
    psi = sqrt(2 nu U_e x / (m + 1)) f(eta),
    U_e = C x^m,   beta = 2m / (m + 1).

``beta = 0`` is the Blasius flat plate (f''(0) = 0.469600, equivalent to the
classical 0.332057 in the variable y sqrt(U/(nu x))), ``beta = 1`` is Hiemenz
stagnation-point flow and ``beta -> -0.19884`` is incipient separation.

All velocities are scaled with a reference speed U_0, x with a length L and y
with L / sqrt(Re_L), so that nu drops out (nu = 1 in scaled units).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_bvp, trapezoid

BETA_SEPARATION = -0.19883768
"""Falkner–Skan pressure-gradient parameter at which f''(0) -> 0."""


@dataclass(frozen=True)
class SimilaritySolution:
    """Falkner–Skan profile f(eta) and its first two derivatives."""

    beta: float
    eta: np.ndarray
    f: np.ndarray
    fp: np.ndarray
    fpp: np.ndarray

    @property
    def m(self) -> float:
        """Exponent of the outer flow U_e = C x^m."""
        return self.beta / (2.0 - self.beta)

    @property
    def wall_shear(self) -> float:
        """f''(0)."""
        return float(self.fpp[0])

    @property
    def displacement_integral(self) -> float:
        """Integral of (1 - f') d eta."""
        return float(trapezoid(1.0 - self.fp, self.eta))

    @property
    def momentum_integral(self) -> float:
        """Integral of f' (1 - f') d eta."""
        return float(trapezoid(self.fp * (1.0 - self.fp), self.eta))

    @property
    def shape_factor(self) -> float:
        """H = delta* / theta."""
        return self.displacement_integral / self.momentum_integral

    def eta_at(self, fraction: float = 0.99) -> float:
        """Smallest eta where f' reaches ``fraction`` (e.g. 0.99 for delta_99)."""
        idx = int(np.argmax(self.fp >= fraction))
        e0, e1 = self.eta[idx - 1], self.eta[idx]
        p0, p1 = self.fp[idx - 1], self.fp[idx]
        return float(e0 + (fraction - p0) * (e1 - e0) / (p1 - p0))

    # ------------------------------------------------------------------
    # Physical fields (scaled units, nu = 1)
    # ------------------------------------------------------------------
    def eta_scale(self, x, ue):
        """Factor g(x) such that eta = y * g(x)."""
        return np.sqrt((self.m + 1.0) * ue / (2.0 * x))

    def velocity(self, x, y, ue):
        """(u, v) at points (x, y) for a local outer velocity ``ue``.

        For a true wedge flow (U_e = C x^m) the result is exact; for other
        U_e it is the *local similarity* approximation.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ue = np.asarray(ue, dtype=float)
        g = self.eta_scale(x, ue)
        eta = y * g
        f, fp = self.interpolate(eta)
        u = ue * fp
        m = self.m
        v = -np.sqrt((m + 1.0) * ue / (2.0 * x)) * (f + (m - 1.0) / (m + 1.0) * eta * fp)
        return u, v

    def interpolate(self, eta):
        """f and f' at arbitrary eta, continued linearly beyond eta_max."""
        eta = np.asarray(eta, dtype=float)
        emax = self.eta[-1]
        f = np.where(eta <= emax, np.interp(eta, self.eta, self.f),
                     self.f[-1] + (eta - emax))
        fp = np.where(eta <= emax, np.interp(eta, self.eta, self.fp), 1.0)
        return f, fp


def _rhs(beta):
    def rhs(_eta, y):
        f, fp, fpp = y
        return np.vstack((fp, fpp, -f * fpp - beta * (1.0 - fp**2)))
    return rhs


def _bc(ya, yb):
    return np.array([ya[0], ya[1], yb[1] - 1.0])


@lru_cache(maxsize=64)
def falkner_skan(beta: float = 0.0, eta_max: float = 10.0, n: int = 2001,
                 tol: float = 1e-10) -> SimilaritySolution:
    """Solve the Falkner–Skan equation for the attached branch.

    Values of ``beta`` close to separation are reached by continuation from
    beta = 0, which keeps the collocation solver on the attached branch.
    """
    if beta < BETA_SEPARATION:
        raise ValueError(f"no attached solution for beta < {BETA_SEPARATION}")

    eta = np.linspace(0.0, eta_max, 201)
    guess = np.vstack((eta - 1.0 + np.exp(-eta), 1.0 - np.exp(-eta), np.exp(-eta)))

    path = [beta] if beta >= -0.1 else list(np.linspace(0.0, beta, 12))
    sol = None
    for b in path:
        sol = solve_bvp(_rhs(b), _bc, eta, guess, tol=tol, max_nodes=200_000)
        if not sol.success:
            raise RuntimeError(f"Falkner–Skan solver failed for beta={b}: {sol.message}")
        eta, guess = sol.x, sol.y

    eta_out = np.linspace(0.0, eta_max, n)
    f, fp, fpp = sol.sol(eta_out)
    return SimilaritySolution(beta=float(beta), eta=eta_out, f=f, fp=fp, fpp=fpp)


def blasius(eta_max: float = 10.0, n: int = 2001) -> SimilaritySolution:
    """Blasius flat-plate solution (beta = 0)."""
    return falkner_skan(0.0, eta_max=eta_max, n=n)


# Classical flat-plate constants in the variable y sqrt(U/(nu x)), i.e. scaled
# by sqrt(2) from the Falkner–Skan form above.
def blasius_constants() -> dict[str, float]:
    """delta_99/x, delta*/x, theta/x and cf, each times sqrt(Re_x)."""
    s = blasius()
    r2 = np.sqrt(2.0)
    return {
        "fpp0_classical": float(s.wall_shear / r2),
        "delta99": float(s.eta_at(0.99) * r2),
        "delta_star": float(s.displacement_integral * r2),
        "theta": float(s.momentum_integral * r2),
        "cf": float(2.0 * s.wall_shear / r2),
        "H": s.shape_factor,
    }
