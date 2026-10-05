"""Validation against classical results from the literature.

References: Blasius (1908), Howarth (1938), Thwaites (1949), White (2006),
Schlichting & Gersten (2017).
"""

import numpy as np
import pytest

from boundary_layer import integral, marching, similarity


def ones(x):
    return np.ones_like(np.asarray(x, dtype=float))


def zeros(x):
    return np.zeros_like(np.asarray(x, dtype=float))


# ----------------------------------------------------------------------
# Similarity solutions
# ----------------------------------------------------------------------
def test_blasius_constants():
    c = similarity.blasius_constants()
    assert c["fpp0_classical"] == pytest.approx(0.332057, abs=2e-6)
    assert c["delta99"] == pytest.approx(4.91, abs=5e-3)
    assert c["delta_star"] == pytest.approx(1.7208, abs=1e-4)
    assert c["theta"] == pytest.approx(0.6641, abs=1e-4)
    assert c["H"] == pytest.approx(2.591, abs=1e-3)


@pytest.mark.parametrize("beta, fpp0", [
    (1.0, 1.232588),      # Hiemenz stagnation point
    (0.5, 0.927680),
    (-0.1, 0.319270),
    (-0.18, 0.128636),
])
def test_falkner_skan_wall_shear(beta, fpp0):
    assert similarity.falkner_skan(beta).wall_shear == pytest.approx(fpp0, abs=1e-5)


def test_falkner_skan_rejects_separated_branch():
    with pytest.raises(ValueError):
        similarity.falkner_skan(-0.25)


def test_blasius_outer_normal_velocity():
    # v sqrt(Re_x)/U -> 0.8604 outside the layer
    s = similarity.blasius()
    _, v = s.velocity(1.0, 20.0, 1.0)
    assert float(v) == pytest.approx(0.8604, abs=1e-3)


# ----------------------------------------------------------------------
# Marching solver
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def flat_plate():
    x = marching.marching_grid(1e-3, 1.0, 300)
    return marching.solve(ones, x, due=zeros)


def test_flat_plate_matches_blasius(flat_plate):
    r = flat_plate
    c = similarity.blasius_constants()
    sq = np.sqrt(r.x[30:])
    # the Rannacher start-up costs < 0.1 % right after x0 and decays downstream
    np.testing.assert_allclose(r.cf[30:] * sq, c["cf"], rtol=1e-3)
    assert r.cf[-1] * sq[-1] == pytest.approx(c["cf"], rel=1e-4)
    np.testing.assert_allclose(r.delta_star[30:] / sq, c["delta_star"], rtol=1e-3)
    np.testing.assert_allclose(r.theta[30:] / sq, c["theta"], rtol=1e-3)
    assert not r.separated


def test_flat_plate_velocity_field(flat_plate):
    r = flat_plate
    ub, vb = similarity.blasius().velocity(r.x[-1], r.y, 1.0)
    assert np.max(np.abs(r.u[-1] - ub)) < 2e-4
    assert np.max(np.abs(r.v[-1] - vb)) < 2e-3


@pytest.mark.parametrize("m", [1.0, 1 / 3, -0.0654])
def test_wedge_flows_match_falkner_skan(m):
    x = marching.marching_grid(0.05, 1.0, 200)
    r = marching.solve(lambda x: np.asarray(x, float) ** m, x,
                       due=lambda x: m * np.asarray(x, float) ** (m - 1))
    s = similarity.falkner_skan(2 * m / (m + 1))
    g = np.sqrt((m + 1) * r.ue / (2 * r.x))
    np.testing.assert_allclose(r.tau_wall / (r.ue * g), s.wall_shear, rtol=1e-3)


def test_howarth_separation():
    # Howarth (1938): x_s/L = 0.1199 for U_e = U_0 (1 - x/L)
    x = marching.marching_grid(1e-3, 0.125, 1200, 1.0)
    r = marching.solve(lambda x: 1 - np.asarray(x, float), x, due=lambda x: -ones(x))
    assert r.separated
    assert r.x_separation == pytest.approx(0.1199, abs=1e-3)


def test_second_order_convergence_in_x():
    def tau(nx):
        x = np.linspace(0.01, 0.1, nx)
        y = marching.wall_grid(12.0, 161)
        r = marching.solve(lambda x: 1 - np.asarray(x, float), x, y,
                           due=lambda x: -ones(x), tol=1e-13)
        return r.tau_wall[-1]

    t1, t2, t3 = tau(41), tau(81), tau(161)
    order = np.log2(abs(t1 - t2) / abs(t2 - t3))
    assert order == pytest.approx(2.0, abs=0.25)


# ----------------------------------------------------------------------
# Integral methods
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name, delta, cf", [
    ("Linear", 3.464, 0.577),
    ("Quadrático", 5.477, 0.730),
    ("Cúbico", 4.641, 0.646),
    ("Quártico", 5.836, 0.685),
    ("Senoidal", 4.795, 0.655),
])
def test_pohlhausen_profiles(name, delta, cf):
    p = integral.profile_by_name(name)
    assert p.delta == pytest.approx(delta, abs=1e-3)
    assert p.cf == pytest.approx(cf, abs=1e-3)


def test_thwaites_flat_plate():
    x = np.linspace(0, 1, 2001)
    t = integral.thwaites(ones, x, due=zeros)
    np.testing.assert_allclose(t.theta[1:] / np.sqrt(x[1:]), np.sqrt(0.45), rtol=1e-9)


def test_thwaites_howarth_separation():
    # closed form: x_s = 1 - (1 + 0.0898/0.075)^(-1/6)
    x = np.linspace(0, 0.2, 40001)
    t = integral.thwaites(lambda x: 1 - x, x, due=lambda x: -ones(x))
    exact = 1 - (1 + 0.0898 / 0.075) ** (-1 / 6)
    assert t.x_separation == pytest.approx(exact, abs=1e-5)


def test_thwaites_stagnation_point():
    # Hiemenz: theta^2 U_e' / nu = 0.075 exactly in Thwaites' closure
    x = np.linspace(0, 1, 1001)
    t = integral.thwaites(lambda x: x, x, due=ones)
    assert t.lam[0] == pytest.approx(0.075)
    # trapezoidal quadrature of x^5 converges quickly away from the origin
    np.testing.assert_allclose(t.lam[100:], 0.075, rtol=1e-3)
