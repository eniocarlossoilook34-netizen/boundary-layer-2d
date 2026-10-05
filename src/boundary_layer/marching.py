"""Space-marching finite-difference solver for the 2D Prandtl equations.

Scaled steady laminar boundary-layer equations (x ~ L, y ~ L/sqrt(Re_L),
u ~ U_0, v ~ U_0/sqrt(Re_L)):

    u_x + v_y = 0
    u u_x + v u_y = U_e dU_e/dx + u_yy

    u(x, 0) = v(x, 0) = 0,   u(x, y -> inf) = U_e(x).

The momentum equation is parabolic in x, so the solution is obtained by
marching downstream from an initial profile. Each step uses a Crank–Nicolson
discretisation centred at x_{n+1/2} (second order in x) and three-point
differences on a stretched wall-normal grid (second order in y). The
nonlinear convective coefficients are converged by Picard iteration, and the
normal velocity at the half step comes straight from integrating continuity:

    v_{n+1/2}(y) = - int_0^y (u_{n+1} - u_n) / dx dy'.

Marching stops at separation (wall shear -> 0), where the Prandtl equations
become singular (Goldstein, 1948).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.linalg import solve_banded

from .similarity import falkner_skan

Array = np.ndarray


# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------
def wall_grid(y_max: float = 12.0, n: int = 301, stretch: float = 3.0) -> Array:
    """Wall-normal grid clustered at the wall: y = y_max sinh(a s) / sinh(a)."""
    s = np.linspace(0.0, 1.0, n)
    if stretch == 0.0:
        return y_max * s
    return y_max * np.sinh(stretch * s) / np.sinh(stretch)


def marching_grid(x0: float, x1: float, n: int = 400, clustering: float = 2.0) -> Array:
    """Streamwise grid clustered near the leading edge: x - x0 ~ s^clustering."""
    s = np.linspace(0.0, 1.0, n)
    return x0 + (x1 - x0) * s**clustering


def _nonuniform_operators(y: Array):
    """Coefficients (lower, centre, upper) of d/dy and d2/dy2 at interior nodes."""
    hm = np.diff(y)[:-1]
    hp = np.diff(y)[1:]
    s = hm + hp
    d1 = (-hp / (hm * s), (hp - hm) / (hm * hp), hm / (hp * s))
    d2 = (2.0 / (hm * s), -2.0 / (hm * hp), 2.0 / (hp * s))
    return d1, d2


def wall_gradient(u: Array, y: Array) -> Array:
    """Second-order one-sided du/dy at y = 0 (works on the last axis)."""
    h1, h2 = y[1] - y[0], y[2] - y[1]
    c0 = -(2 * h1 + h2) / (h1 * (h1 + h2))
    c1 = (h1 + h2) / (h1 * h2)
    c2 = -h1 / (h2 * (h1 + h2))
    return c0 * u[..., 0] + c1 * u[..., 1] + c2 * u[..., 2]


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class MarchingResult:
    """Boundary-layer solution on the (x, y) grid, in scaled units."""

    x: Array
    y: Array
    u: Array            # shape (nx, ny)
    v: Array            # shape (nx, ny)
    ue: Array
    iterations: Array
    separated: bool = False
    x_separation: float | None = None
    info: dict = field(default_factory=dict)

    @property
    def tau_wall(self) -> Array:
        """Scaled wall shear du/dy|_0 (= tau_w L / (mu U_0 sqrt(Re_L)))."""
        return wall_gradient(self.u, self.y)

    @property
    def cf(self) -> Array:
        """Skin-friction coefficient times sqrt(Re_L)."""
        return 2.0 * self.tau_wall / self.ue**2

    @property
    def delta_star(self) -> Array:
        return trapezoid(1.0 - self.u / self.ue[:, None], self.y, axis=1)

    @property
    def theta(self) -> Array:
        r = self.u / self.ue[:, None]
        return trapezoid(r * (1.0 - r), self.y, axis=1)

    @property
    def shape_factor(self) -> Array:
        return self.delta_star / self.theta

    def delta(self, fraction: float = 0.99) -> Array:
        """Thickness where u/U_e first reaches ``fraction``."""
        out = np.empty(len(self.x))
        for i, (row, ue) in enumerate(zip(self.u, self.ue)):
            r = row / ue
            k = int(np.argmax(r >= fraction))
            out[i] = np.interp(fraction, r[k - 1:k + 1], self.y[k - 1:k + 1])
        return out

    def vorticity(self) -> Array:
        """Boundary-layer vorticity -du/dy (scaled)."""
        return -np.gradient(self.u, self.y, axis=1)


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------
def similarity_profile(x0: float, ue0: float, m: float, y: Array) -> tuple[Array, Array]:
    """Falkner–Skan profile with local exponent m, used as initial condition."""
    beta = 2.0 * m / (m + 1.0)
    sol = falkner_skan(round(beta, 12))
    return sol.velocity(x0, y, ue0)


def solve(
    ue: Callable[[Array], Array],
    x: Array,
    y: Array | None = None,
    due: Callable[[Array], Array] | None = None,
    u0: Array | None = None,
    v0: Array | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    tau_min: float = 1e-3,
    startup_steps: int = 2,
) -> MarchingResult:
    """March the Prandtl equations from x[0] to x[-1].

    Parameters
    ----------
    ue, due
        Outer velocity U_e(x) and its derivative. ``due`` defaults to a
        centred finite difference of ``ue``.
    x, y
        Marching stations and wall-normal grid (``y[0]`` must be 0).
    u0, v0
        Initial profile at x[0]. Defaults to the Falkner–Skan profile whose
        exponent m = x U_e'/U_e matches the local outer flow.
    tau_min
        Marching stops when the scaled wall shear drops below this value,
        and the separation point is extrapolated from tau_w^2 (which is
        linear in x near a Goldstein singularity).
    startup_steps
        Number of initial intervals advanced with two implicit-Euler half
        steps instead of Crank–Nicolson (Rannacher, 1984).
    """
    x = np.asarray(x, dtype=float)
    y = wall_grid() if y is None else np.asarray(y, dtype=float)
    if due is None:
        def due(xx, _h=1e-6):
            return (ue(xx + _h) - ue(xx - _h)) / (2 * _h)

    ue_x = np.asarray(ue(x), dtype=float) * np.ones_like(x)
    if u0 is None:
        m0 = float(x[0] * due(x[0]) / ue_x[0])
        u0, v0 = similarity_profile(x[0], ue_x[0], m0, y)
    if v0 is None:
        v0 = np.zeros_like(y)

    nx, ny = len(x), len(y)
    U = np.zeros((nx, ny))
    V = np.zeros((nx, ny))
    its = np.zeros(nx, dtype=int)
    U[0], V[0] = u0, v0

    (a1, b1, c1), (a2, b2, c2) = _nonuniform_operators(y)
    ab = np.zeros((3, ny - 2))

    def step(un, xa, xb, theta, guess=None):
        """One implicit step x_a -> x_b; theta = 1/2 is Crank–Nicolson, 1 is Euler."""
        dx = xb - xa
        xm = xa + theta * dx
        pressure = float(ue(xm) * due(xm))
        u = un.copy() if guess is None else guess.copy()
        u[0], u[-1] = 0.0, float(ue(xb))
        k, converged = 0, False
        for k in range(max_iter):
            ubar = theta * u + (1 - theta) * un
            vbar = -cumulative_trapezoid((u - un) / dx, y, initial=0.0)
            ui, vi = ubar[1:-1], vbar[1:-1]
            lo_op, di_op, up_op = vi * a1 - a2, vi * b1 - b2, vi * c1 - c2
            rhs = (pressure + ui * un[1:-1] / dx - (1 - theta) * (
                lo_op * un[:-2] + di_op * un[1:-1] + up_op * un[2:]))
            # Dirichlet conditions u_0 = 0, u_N = U_e(x_b)
            rhs[-1] -= theta * up_op[-1] * u[-1]
            ab[0, 1:] = theta * up_op[:-1]
            ab[1] = ui / dx + theta * di_op
            ab[2, :-1] = theta * lo_op[1:]
            new = solve_banded((1, 1), ab, rhs)
            change = np.max(np.abs(new - u[1:-1]))
            u[1:-1] = new
            if change < tol:
                converged = True
                break
        return u, converged, k + 1

    separated, x_sep = False, None
    last = nx - 1
    for n in range(nx - 1):
        un = U[n]
        if n < startup_steps:
            # Rannacher start-up: two implicit-Euler half steps damp the
            # high-frequency error that Crank–Nicolson leaves undamped when
            # the initial profile is not an exact solution.
            xh = 0.5 * (x[n] + x[n + 1])
            uh, c1_, k1 = step(un, x[n], xh, 1.0)
            u, c2_, k2 = step(uh, xh, x[n + 1], 1.0)
            converged, k = c1_ and c2_, k1 + k2
        else:
            # linear extrapolation from the two previous stations as first guess
            r = (x[n + 1] - x[n]) / (x[n] - x[n - 1])
            guess = un + r * (un - U[n - 1])
            u, converged, k = step(un, x[n], x[n + 1], 0.5, guess)
        dx = x[n + 1] - x[n]

        U[n + 1] = u
        V[n + 1] = -cumulative_trapezoid((u - un) / dx, y, initial=0.0)
        its[n + 1] = k

        tau = wall_gradient(u, y)
        if tau < tau_min or not converged or np.any(u[1:-1] < 0):
            separated = True
            tw = wall_gradient(U[:n + 1], y)
            # tau_w^2 ~ (x_s - x) near a Goldstein singularity: fit the last stations
            sel = slice(max(0, n - 5), n + 1)
            p = np.polyfit(x[sel], tw[sel] ** 2, 1)
            x_sep = float(-p[1] / p[0])
            last = n          # discard the station that failed
            break

    # v at the stations from second-order differences of u (continuity)
    if last >= 2:
        dudx = np.gradient(U[:last + 1], x[:last + 1], axis=0, edge_order=2)
        V[:last + 1] = -cumulative_trapezoid(dudx, y, axis=1, initial=0.0)
    V[0] = v0

    sl = slice(0, last + 1)
    return MarchingResult(
        x=x[sl], y=y, u=U[sl], v=V[sl], ue=ue_x[sl], iterations=its[sl],
        separated=separated, x_separation=x_sep,
        info={"tol": tol, "nx": nx, "ny": ny},
    )
