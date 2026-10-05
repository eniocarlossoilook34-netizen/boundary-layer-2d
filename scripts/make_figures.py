"""Generate every figure in ``figures/``.

    python scripts/make_figures.py            # all figures
    python scripts/make_figures.py 1 5        # only figures 1 and 5
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import NullFormatter
from scipy.integrate import cumulative_trapezoid

from boundary_layer import integral, marching, similarity, turbulent
from boundary_layer import plotting as P

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
P.use_style()

# Physical case used whenever dimensions are shown: air at 20 °C over a 1 m plate
NU = 1.5e-5        # m^2/s
U_INF = 5.0        # m/s
L = 1.0            # m
RE_L = U_INF * L / NU
Y_MM = 1e3 * L / np.sqrt(RE_L)     # scaled y -> mm
RHO = 1.2


def ones(x):
    return np.ones_like(np.asarray(x, dtype=float))


def zeros(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def save(fig, name):
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(OUT.parent)}")


_cache: dict = {}


def flat_plate():
    if "flat" not in _cache:
        x = marching.marching_grid(1e-3, 1.0, 500)
        y = marching.wall_grid(12.0, 321, 3.0)
        _cache["flat"] = marching.solve(ones, x, y, due=zeros)
    return _cache["flat"]


def howarth():
    if "howarth" not in _cache:
        x = marching.marching_grid(1e-3, 0.125, 1500, 1.0)
        y = marching.wall_grid(14.0, 321, 3.0)
        _cache["howarth"] = marching.solve(lambda x: 1 - np.asarray(x, float), x, y,
                                           due=lambda x: -ones(x))
    return _cache["howarth"]


def stream_function(res):
    return cumulative_trapezoid(res.u, res.y, axis=1, initial=0.0)


# ----------------------------------------------------------------------
# 01 - hero: the u field with theory and numerics on the same plane
# ----------------------------------------------------------------------
def fig01():
    r = flat_plate()
    bl = similarity.blasius()
    cst = similarity.blasius_constants()
    cub = integral.profile_by_name("Cúbico")
    xt = np.linspace(1e-3, 1.0, 2000)
    thw = integral.thwaites(ones, np.linspace(0, 1, 4001)[1:], due=zeros)

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])

    X, Yg = np.meshgrid(r.x, r.y * Y_MM, indexing="ij")
    ymax = 14.0
    cf = ax.contourf(X, Yg, np.clip(r.u, 0, 1), levels=np.linspace(0, 1, 21), cmap=P.SEQUENTIAL)
    psi = stream_function(r)
    ax.contour(X, Yg, psi, levels=np.linspace(0.25, 8.0, 14), colors="white",
               linewidths=0.6, alpha=0.55)

    sq = np.sqrt(xt)
    ax.plot(r.x, r.delta() * Y_MM, color=P.ORANGE, lw=2.6,
            label=r"$\delta_{99}$ numérico (Crank–Nicolson)")
    ax.plot(xt, cst["delta99"] * sq * Y_MM, color="white", lw=1.6, ls=(0, (5, 3)),
            label=r"$\delta_{99}$ Blasius: $4.91\,x/\sqrt{Re_x}$")
    ax.plot(xt, cub.delta * sq * Y_MM, color=P.YELLOW, lw=2.0, ls=(0, (1, 1.6)),
            label=r"$\delta$ Pohlhausen cúbico: $4.64\,x/\sqrt{Re_x}$")
    ax.plot(r.x, r.delta_star * Y_MM, color="white", lw=1.2, ls="-.",
            label=r"$\delta^*$ numérico (deslocamento)")
    ax.plot(thw.x, thw.delta_star * Y_MM, color=P.AQUA, lw=1.4, ls=(0, (3, 1, 1, 1)),
            label=r"$\delta^*$ Thwaites")

    # velocity profiles drawn on the plane
    for xs in (0.15, 0.4, 0.65, 0.85):
        i = int(np.argmin(np.abs(r.x - xs)))
        scale = 0.13
        dloc = r.delta()[i] * Y_MM
        ax.plot([r.x[i]] * 2, [0, ymax], color="white", lw=0.6, alpha=0.6)
        ax.plot(r.x[i] + scale * r.u[i], r.y * Y_MM, color=P.INK, lw=1.6)
        for yy in np.linspace(0.12, 1.25, 6) * dloc:
            uu = np.interp(yy / Y_MM, r.y, r.u[i])
            if uu > 0.06:
                ax.annotate("", (r.x[i] + scale * uu, yy), (r.x[i], yy),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=P.INK,
                                            shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(r.x[i] + 0.005, ymax - 0.6, f"x = {r.x[i]:.2f} m", color="white",
                fontsize=8.5, va="top")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, ymax)
    ax.grid(False)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [mm]")
    mant, expo = f"{RE_L:.2e}".split("e+")
    ax.set_title(f"Camada limite sobre placa plana — ar, $U_\\infty$ = {U_INF:.0f} m/s, "
                 f"L = {L:.0f} m, $Re_L = {mant}\\times10^{{{int(expo)}}}$", pad=22)
    ax.text(0.0, 1.015, "Campo $u/U_\\infty$ do solver numérico; curvas: teoria e métodos "
            "aproximados sobre o mesmo plano.  Escala vertical ampliada ≈ 70×.",
            transform=ax.transAxes, fontsize=9, color=P.INK_2, va="bottom")
    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.03, ticks=np.linspace(0, 1, 6))
    cbar.set_label(r"$u/U_\infty$", color=P.INK_2)
    cbar.outline.set_visible(False)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3, fontsize=8.8,
                    handlelength=3.2)
    for t in leg.get_texts():
        t.set_color(P.INK_2)
    # legend sits on a light background, so swap white lines for ink in the legend
    for h in leg.legend_handles:
        if h.get_color() == "white":
            h.set_color(P.INK_3)

    # (b) self-similarity
    axb = fig.add_subplot(gs[1, 0])
    eta = np.linspace(0, 8, 400)
    axb.plot(bl.interpolate(eta)[1], eta * np.sqrt(2), color=P.BLUE, lw=2, label="Blasius (exato)")
    markers = ["o", "s", "^"]
    for k, xs in enumerate((0.05, 0.3, 1.0)):
        i = int(np.argmin(np.abs(r.x - xs)))
        e = r.y / np.sqrt(r.x[i])
        sel = e < 8 * np.sqrt(2)
        idx = np.nonzero(sel)[0][::9]
        axb.plot(r.u[i, idx], e[idx], ls="none", marker=markers[k], ms=5, mfc="none",
                 mec=P.ORANGE, mew=1.2, label=f"numérico, x = {r.x[i]:.2f}")
    axb.set_xlabel(r"$u/U_\infty$")
    axb.set_ylabel(r"$y\sqrt{U_\infty/\nu x}$")
    axb.set_title("Autossimilaridade dos perfis")
    axb.set_ylim(0, 8)
    axb.legend(loc="upper left")

    # (c) wall shear stress
    axc = fig.add_subplot(gs[1, 1])
    tau_scale = RHO * U_INF**2 / np.sqrt(RE_L)    # scaled shear -> Pa
    axc.plot(xt, cst["fpp0_classical"] / sq * tau_scale, color=P.BLUE, lw=2, label="Blasius")
    axc.plot(thw.x, thw.tau_wall * tau_scale, color=P.AQUA, lw=1.6, ls="--", label="Thwaites")
    axc.plot(xt, cub.cf / 2 / sq * tau_scale, color=P.YELLOW, lw=1.6, ls=":",
             label="Pohlhausen cúbico")
    idx = np.unique(np.geomspace(1, len(r.x) - 1, 22).astype(int))
    axc.plot(r.x[idx], r.tau_wall[idx] * tau_scale, ls="none", marker="o", ms=5,
             mfc="none", mec=P.ORANGE, mew=1.2, label="numérico")
    axc.set_xlabel("x  [m]")
    axc.set_ylabel(r"$\tau_w$  [Pa]")
    axc.set_title("Cisalhamento na parede")
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 0.06)
    axc.legend(loc="upper right")

    # (d) error field
    axd = fig.add_subplot(gs[1, 2])
    ub, _ = bl.velocity(X, Yg / Y_MM, 1.0)
    err = np.abs(r.u - ub)
    err = np.clip(err, 1e-7, None)
    pm = axd.pcolormesh(X, Yg, err, cmap=P.SEQUENTIAL, norm=LogNorm(1e-6, 1e-3),
                        shading="gouraud", rasterized=True)
    axd.plot(r.x, r.delta() * Y_MM, color=P.ORANGE, lw=1.4)
    axd.set_ylim(0, ymax)
    axd.set_xlim(0, 1)
    axd.grid(False)
    axd.set_xlabel("x  [m]")
    axd.set_ylabel("y  [mm]")
    axd.set_title(r"Erro $|u_{num} - u_{Blasius}|/U_\infty$")
    cb = fig.colorbar(pm, ax=axd, pad=0.02, fraction=0.06)
    cb.outline.set_visible(False)
    axd.text(0.97, 0.06, f"máx = {err.max():.1e}", transform=axd.transAxes, ha="right",
             fontsize=8.5, color="white")

    save(fig, "01_placa_plana_campo_u.png")


# ----------------------------------------------------------------------
# 02 - Blasius: f, f', f'' and v
# ----------------------------------------------------------------------
def fig02():
    bl = similarity.blasius()
    r = flat_plate()
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.28})

    e = bl.eta
    sel = e <= 7
    a.plot(bl.f[sel], e[sel], color=P.BLUE, label=r"$f$  (função de corrente)")
    a.plot(bl.fp[sel], e[sel], color=P.ORANGE, ls="--", label=r"$f' = u/U_e$")
    a.plot(bl.fpp[sel] / bl.fpp[0], e[sel], color=P.AQUA, ls=":", lw=2.4,
           label=r"$f''/f''(0)$  (cisalhamento)")
    a.axhline(bl.eta_at(0.99), color=P.INK_3, lw=0.8)
    a.text(1.05, bl.eta_at(0.99) + 0.1, r"$\eta_{99} = 3.472$", color=P.INK_2, fontsize=9)
    a.set_xlim(0, 3.5)
    a.set_ylim(0, 7)
    a.set_xlabel(r"$f,\ f',\ f''/f''(0)$")
    a.set_ylabel(r"$\eta = y\sqrt{U_\infty / 2\nu x}$")
    a.set_title(r"Blasius: $f''' + f f'' = 0$,   $f''(0) = 0.46960$")
    a.legend(loc="upper right")

    # v is the part of the solution the similarity variables hide best
    eta = np.linspace(0, 7, 300)
    f, fp = bl.interpolate(eta)
    vb = (eta * fp - f) / np.sqrt(2)      # v sqrt(Re_x) / U  = (eta f' - f)/sqrt(2)
    b.plot(vb, eta * np.sqrt(2), color=P.BLUE, label="Blasius (exato)")
    for k, xs in enumerate((0.05, 0.3, 1.0)):
        i = int(np.argmin(np.abs(r.x - xs)))
        er = r.y / np.sqrt(r.x[i])
        idx = np.nonzero(er < 7 * np.sqrt(2))[0][::9]
        b.plot(r.v[i, idx] * np.sqrt(r.x[i]), er[idx], ls="none", marker="osd"[k], ms=5,
               mfc="none", mec=P.ORANGE, mew=1.2, label=f"numérico, x = {r.x[i]:.2f}")
    b.axvline(0.8604, color=P.INK_3, lw=0.8)
    b.text(0.845, 0.5, r"$v_\infty\sqrt{Re_x}/U = 0.8604$", color=P.INK_2, fontsize=9,
           ha="right")
    b.set_xlabel(r"$v\,\sqrt{Re_x}\,/\,U_\infty$")
    b.set_ylabel(r"$y\sqrt{U_\infty/\nu x}$")
    b.set_ylim(0, 9.5)
    b.set_title("Velocidade normal: continuidade integrada pelo solver")
    b.legend(loc="upper left")
    save(fig, "02_blasius_similaridade.png")


# ----------------------------------------------------------------------
# 03 - Falkner–Skan family vs marching solver on wedge flows
# ----------------------------------------------------------------------
def fig03():
    betas = [similarity.BETA_SEPARATION + 1e-6, -0.14, 0.0, 0.3, 1.0, 2.0]
    norm = Normalize(-0.6, 0.6)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"wspace": 0.3})
    a, b, c = axs
    for beta in betas:
        s = similarity.falkner_skan(beta)
        # adverse (beta < 0) -> red arm, favourable -> blue arm, Blasius -> neutral ink
        t = np.clip(-beta * 2.6 if beta < 0 else -0.2 - beta * 0.2, -0.6, 0.6)
        col = P.INK_2 if beta == 0.0 else P.DIVERGING(norm(t))
        sel = s.eta <= 5
        txt = r"$\beta$ = " + (f"{beta:.4f} (separação)" if beta < -0.19 else
                                f"{beta:g}" + ("  (Blasius)" if beta == 0 else
                                               "  (Hiemenz)" if beta == 1 else ""))
        a.plot(s.fp[sel], s.eta[sel], color=col, lw=2, label=txt)
    a.set_xlabel(r"$f' = u/U_e$")
    a.set_ylabel(r"$\eta$")
    a.set_ylim(0, 5)
    a.set_xlim(0, 1.15)
    a.set_title(r"Perfis de Falkner–Skan,  $U_e = Cx^m$")
    a.legend(loc="upper left", fontsize=8.5, title="vermelho: adverso · azul: favorável",
             title_fontsize=8.5, alignment="left")

    bs = np.concatenate([np.linspace(similarity.BETA_SEPARATION + 1e-5, -0.15, 12),
                         np.linspace(-0.14, 2.0, 40)])
    sols = [similarity.falkner_skan(float(bb), n=1201) for bb in bs]
    b.plot(bs, [s.wall_shear for s in sols], color=P.BLUE, label="Falkner–Skan (exato)")
    c.plot(bs, [s.shape_factor for s in sols], color=P.BLUE, label="Falkner–Skan (exato)")

    # marching solver on real wedge flows: start at x=0.05, compare at x=1
    y = marching.wall_grid(14.0, 321, 3.0)
    bnum, tnum, hnum = [], [], []
    for beta in (-0.19, -0.16, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0):
        m = beta / (2 - beta)
        x = marching.marching_grid(0.05, 1.0, 300)
        r = marching.solve(lambda xx, m=m: np.asarray(xx, float) ** m, x, y,
                           due=lambda xx, m=m: m * np.asarray(xx, float) ** (m - 1))
        g = np.sqrt((m + 1) * r.ue[-1] / (2 * r.x[-1]))
        bnum.append(beta)
        tnum.append(r.tau_wall[-1] / (r.ue[-1] * g))
        hnum.append(r.shape_factor[-1])
    for ax, val in ((b, tnum), (c, hnum)):
        ax.plot(bnum, val, ls="none", marker="o", ms=7, mfc="none", mec=P.ORANGE, mew=1.5,
                label="solver de marcha em $U_e = x^m$")
        ax.axvline(similarity.BETA_SEPARATION, color=P.INK_3, lw=0.8, ls="--")
        ax.set_xlabel(r"$\beta = 2m/(m+1)$")
        ax.legend(loc="upper right" if ax is c else "lower right")
    b.text(similarity.BETA_SEPARATION + 0.04, 1.45, "separação\n" + r"$\beta = -0.1988$",
           fontsize=8.5, color=P.INK_2, va="top")
    b.set_ylabel(r"$f''(0)$")
    b.set_title("Cisalhamento na parede")
    c.set_ylabel(r"$H = \delta^*/\theta$")
    c.set_title("Fator de forma")
    save(fig, "03_falkner_skan.png")


# ----------------------------------------------------------------------
# 04 - Integral methods: assumed profiles and an error table
# ----------------------------------------------------------------------
def fig04():
    bl = similarity.blasius()
    cst = similarity.blasius_constants()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5),
                               gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.25})

    s = np.linspace(0, 1.25, 300)
    eta99 = bl.eta_at(0.99)
    a.plot(bl.interpolate(s * eta99)[1], s, color=P.INK, lw=2.6, label="Blasius (exato)")
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
    cols = [P.BLUE, P.ORANGE, P.AQUA, P.YELLOW, P.MAGENTA]
    for p, ls, col in zip(integral.PROFILES, styles, cols):
        # place each assumed profile on its own predicted thickness relative to delta_99
        a.plot(p.profile(s * cst["delta99"] / p.delta), s, color=col, ls=ls, lw=1.6,
               label=f"{p.name}: {p.latex}")
    a.set_xlabel(r"$u/U_\infty$")
    a.set_ylabel(r"$y/\delta_{99,\,Blasius}$")
    a.set_ylim(0, 1.25)
    a.set_xlim(0, 1.05)
    a.set_title("Perfis admitidos (von Kármán–Pohlhausen)")
    a.legend(loc="upper left", fontsize=8.3)

    rows = [*(p.name for p in integral.PROFILES), "Thwaites"]
    cols_t = [r"$\delta/x$", r"$\delta^*/x$", r"$\theta/x$", r"$c_f$", r"$H$"]
    ref = np.array([cst["delta99"], cst["delta_star"], cst["theta"], cst["cf"], cst["H"]])
    vals = [[p.delta, p.delta_star, p.theta, p.cf, p.H] for p in integral.PROFILES]
    th = np.sqrt(0.45)
    lam0 = 0.0
    vals.append([np.nan, float(integral.thwaites_H(lam0)) * th, th,
                 2 * float(integral.thwaites_l(lam0)) / th, float(integral.thwaites_H(lam0))])
    vals = np.array(vals)
    err = 100 * (vals - ref) / ref
    im = b.imshow(err, cmap=P.DIVERGING, vmin=-40, vmax=40, aspect="auto")
    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            if np.isnan(err[i, j]):
                b.text(j, i, "—", ha="center", va="center", color=P.INK_3)
                continue
            dark = abs(err[i, j]) > 22
            b.text(j, i, f"{vals[i, j]:.3f}\n{err[i, j]:+.1f}%", ha="center", va="center",
                   fontsize=8.5, color="white" if dark else P.INK, linespacing=1.3)
    b.set_xticks(range(len(cols_t)), [c + r"$\ \sqrt{Re_x}$" if k < 4 else c
                                       for k, c in enumerate(cols_t)])
    b.set_yticks(range(len(rows)), rows)
    b.tick_params(length=0)
    b.grid(False)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_xticks(np.arange(-0.5, len(cols_t)), minor=True)
    b.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    b.grid(which="minor", color=P.SURFACE, lw=2)
    b.set_title("Resultado e erro relativo a Blasius  " +
                f"(ref.: {ref[0]:.3f}, {ref[1]:.4f}, {ref[2]:.4f}, {ref[3]:.4f}, {ref[4]:.3f})",
                fontsize=10)
    cb = fig.colorbar(im, ax=b, fraction=0.04, pad=0.02)
    cb.set_label("erro [%]", color=P.INK_2)
    cb.outline.set_visible(False)
    save(fig, "04_metodos_integrais.png")


# ----------------------------------------------------------------------
# 05 - Howarth's decelerating flow: separation
# ----------------------------------------------------------------------
X_SEP_HOWARTH = 0.1199     # Howarth (1938); Smith (1977): 0.11988


def fig05():
    r = howarth()
    xt = np.linspace(0, 0.13, 20001)
    th = integral.thwaites(lambda x: 1 - x, xt, due=lambda x: -ones(x))

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.42, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    X, Yg = np.meshgrid(r.x, r.y, indexing="ij")
    cf = ax.contourf(X, Yg, np.clip(r.u / r.ue[:, None], 0, 1), levels=np.linspace(0, 1, 21),
                     cmap=P.SEQUENTIAL)
    psi = stream_function(r)
    ax.contour(X, Yg, psi, levels=np.geomspace(0.01, 4, 14), colors="white",
               linewidths=0.6, alpha=0.6)
    ax.plot(r.x, r.delta(), color=P.ORANGE, lw=2.4, label=r"$\delta_{99}$ numérico")
    ax.plot(r.x, r.delta_star, color="white", lw=1.2, ls="-.", label=r"$\delta^*$ numérico")
    ax.plot(th.x, th.delta_star, color=P.AQUA, lw=1.6, ls=(0, (3, 1, 1, 1)),
            label=r"$\delta^*$ Thwaites")
    for xs in (0.02, 0.06, 0.1):
        i = int(np.argmin(np.abs(r.x - xs)))
        scale = 0.012
        ax.plot([r.x[i]] * 2, [0, 4.5], color="white", lw=0.5, alpha=0.5)
        ax.plot(r.x[i] + scale * r.u[i] / r.ue[i], r.y, color=P.INK, lw=1.6)
    ax.axvline(X_SEP_HOWARTH, color=P.RED, lw=1.4, ls="--")
    ax.text(X_SEP_HOWARTH - 0.0012, 4.3, "separação (Howarth, 1938)\n"
            f"x/L = {X_SEP_HOWARTH}", color="white", ha="right", va="top", fontsize=9)
    ax.set_xlim(0, 0.13)
    ax.set_ylim(0, 4.5)
    ax.grid(False)
    ax.set_xlabel(r"$x/L$")
    ax.set_ylabel(r"$y\sqrt{Re_L}/L$")
    ax.set_title(r"Escoamento desacelerado de Howarth,  $U_e = U_0(1 - x/L)$:  "
                 r"o gradiente adverso conduz à separação")
    cbar = fig.colorbar(cf, ax=ax, pad=0.012, fraction=0.03, ticks=np.linspace(0, 1, 6))
    cbar.set_label(r"$u/U_e$", color=P.INK_2)
    cbar.outline.set_visible(False)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3)
    for h in leg.legend_handles:
        if h.get_color() == "white":
            h.set_color(P.INK_3)

    a = fig.add_subplot(gs[1, 0])
    a.plot(r.x, r.tau_wall, color=P.ORANGE, lw=2, label="numérico (CN)")
    a.plot(th.x, th.tau_wall, color=P.AQUA, lw=1.8, ls="--", label="Thwaites")
    a.plot(xt[1:], 0.332 / np.sqrt(xt[1:]) * (1 - xt[1:]) ** 1.5, color=P.INK_3, lw=1.2,
           ls=":", label="Blasius local (sem $dp/dx$)")
    a.axvline(X_SEP_HOWARTH, color=P.RED, lw=1, ls="--")
    a.plot([r.x_separation], [0], marker="o", color=P.ORANGE, ms=8, mec=P.SURFACE, mew=2,
           clip_on=False, zorder=5)
    a.plot([th.x_separation], [0], marker="D", color=P.AQUA, ms=7, mec=P.SURFACE, mew=2,
           clip_on=False, zorder=5)
    a.set_ylim(0, 4)
    a.set_xlim(0, 0.13)
    a.set_xlabel(r"$x/L$")
    a.set_ylabel(r"$\tau_w L/(\mu U_0\sqrt{Re_L})$")
    a.set_title("Cisalhamento na parede")
    a.legend(loc="upper right", fontsize=8.5)
    a.text(0.40, 0.45, f"numérico: $x_s$ = {r.x_separation:.4f}\n"
           f"Thwaites: $x_s$ = {th.x_separation:.4f}", transform=a.transAxes, fontsize=8.5,
           color=P.INK_2)

    b = fig.add_subplot(gs[1, 1])
    stations = [0.0, 0.04, 0.08, 0.10, 0.11, r.x[-1]]
    ramp = [P.BLUE_RAMP[k] for k in (3, 5, 7, 9, 11, 12)]
    for xs, col in zip(stations, ramp):
        i = int(np.argmin(np.abs(r.x - xs)))
        b.plot(r.u[i] / r.ue[i], r.y, color=col, lw=1.8, label=f"x/L = {r.x[i]:.3f}")
    b.legend(loc="upper left", fontsize=8.5)
    b.set_xlim(0, 1.05)
    b.set_ylim(0, 6)
    b.set_xlabel(r"$u/U_e$")
    b.set_ylabel(r"$y\sqrt{Re_L}/L$")
    b.set_title(r"Perfis: $\partial u/\partial y|_w \to 0$ e ponto de inflexão")

    c = fig.add_subplot(gs[1, 2])
    c.plot(r.x, r.shape_factor, color=P.ORANGE, lw=2, label="numérico (CN)")
    c.plot(th.x, th.H, color=P.AQUA, lw=1.8, ls="--", label="Thwaites")
    c.axhline(2.591, color=P.INK_3, lw=0.8)
    c.text(0.07, 2.535, "Blasius: 2.591", fontsize=8.5, color=P.INK_2)
    c.set_ylim(2.5, 3.9)
    c.axvline(X_SEP_HOWARTH, color=P.RED, lw=1, ls="--")
    c.set_xlim(0, 0.13)
    c.set_xlabel(r"$x/L$")
    c.set_ylabel(r"$H = \delta^*/\theta$")
    c.set_title("Fator de forma")
    c.legend(loc="upper left")
    save(fig, "05_howarth_separacao.png")


# ----------------------------------------------------------------------
# 06 - convergence of the marching scheme
# ----------------------------------------------------------------------
X0_CONV, X1_CONV = 0.01, 0.1


def _howarth_tau(nx, ny, startup=2):
    x = np.linspace(X0_CONV, X1_CONV, nx)
    y = marching.wall_grid(12.0, ny, 3.0)
    r = marching.solve(lambda x: 1 - np.asarray(x, float), x, y, due=lambda x: -ones(x),
                       tol=1e-13, startup_steps=startup)
    return r.tau_wall[-1]


def convergence_study():
    nxs = np.array([21, 41, 81, 161, 321])
    nys = np.array([41, 81, 161, 321])
    out = {"dx": (X1_CONV - X0_CONV) / (nxs - 1),
           "dy": np.array([marching.wall_grid(12.0, n, 3.0)[1] for n in nys])}
    for su, key in ((2, "ex"), (0, "ex_cn")):
        ref = _howarth_tau(1281, 321, su)
        out[key] = np.array([abs(_howarth_tau(n, 321, su) - ref) / ref for n in nxs])
    ref = _howarth_tau(321, 1281)
    out["ey"] = np.array([abs(_howarth_tau(321, n) - ref) / ref for n in nys])
    return out


def _order(h, e):
    return np.polyfit(np.log(h[-3:]), np.log(e[-3:]), 1)[0]


def fig06():
    d = convergence_study()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"wspace": 0.3})
    h = d["dx"]
    local = np.diff(np.log(d["ex_cn"])) / np.diff(np.log(h))
    lo_cn, hi_cn = local.min(), local.max()
    a.loglog(h, d["ex"], marker="o", color=P.BLUE, ms=7, mec=P.SURFACE, mew=1.5,
             label=f"CN + partida de Rannacher (ordem {_order(h, d['ex']):.2f})")
    a.loglog(h, d["ex_cn"], marker="s", color=P.ORANGE, ms=7, mec=P.SURFACE, mew=1.5,
             ls="--", label=f"CN puro (ordem local {lo_cn:.1f} a {hi_cn:.1f}, irregular)")
    a.loglog(h, d["ex"][-1] * (h / h[-1]) ** 2, color=P.INK_3, lw=1, ls=":",
             label="referência: ordem 2")
    a.set_xlabel(r"$\Delta x / L$")
    a.set_title("Refinamento em x")
    hy = d["dy"]
    b.loglog(hy, d["ey"], marker="o", color=P.BLUE, ms=7, mec=P.SURFACE, mew=1.5,
             label=f"diferenças em malha esticada (ordem {_order(hy, d['ey']):.2f})")
    b.loglog(hy, d["ey"][-1] * (hy / hy[-1]) ** 2, color=P.INK_3, lw=1, ls=":",
             label="referência: ordem 2")
    b.set_xlabel(r"$\Delta y_{parede}\,\sqrt{Re_L}/L$")
    b.set_title("Refinamento em y")
    b.xaxis.set_minor_formatter(NullFormatter())
    for ax in (a, b):
        ax.set_ylabel(r"erro relativo em $\tau_w$ ($x/L = 0.1$)")
        ax.legend(loc="upper left", fontsize=8.8)
        ax.grid(True, which="both", lw=0.4)
    fig.suptitle("Escoamento de Howarth, marcha de x/L = 0.01 a 0.1 — erro contra solução "
                 "em malha 4× mais fina", x=0.06, ha="left", fontsize=9.5, color=P.INK_2, y=0.01)
    save(fig, "06_convergencia.png")
    return d


# ----------------------------------------------------------------------
# 07 - cf over the full Reynolds range
# ----------------------------------------------------------------------
def fig07():
    r = flat_plate()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    re_lam = np.geomspace(1e3, 3e6, 200)
    re_tur = np.geomspace(1e5, 1e8, 200)
    ax.axvspan(3e5, 3e6, color=P.GRID, alpha=0.6, lw=0)
    ax.text(9.5e5, 1.2e-3, "faixa típica\nde transição", ha="center", fontsize=8.5,
            color=P.INK_2)
    ax.text(1.1e3, 4.2e-4, "teoria de camada limite\n" r"exige $Re_x \gg 1$", fontsize=8.5,
            color=P.INK_2)
    ax.loglog(re_lam, turbulent.cf_laminar(re_lam), color=P.BLUE,
              label=r"laminar, Blasius: $0.664/\sqrt{Re_x}$")
    ax.loglog(re_tur, turbulent.cf_power_law(re_tur), color=P.ORANGE, ls="--",
              label=r"turbulento, lei 1/7: $0.0592\,Re_x^{-1/5}$")
    ax.loglog(re_tur, turbulent.cf_white(re_tur), color=P.AQUA, ls="-.",
              label=r"turbulento, White: $0.455/\ln^2(0.06\,Re_x)$")
    # the scaled solution is valid for any Re: rescale to three plate Reynolds numbers
    for k, rel in enumerate((1e4, 1e5, 1e6)):
        idx = np.unique(np.geomspace(10, len(r.x) - 1, 8).astype(int))
        rex = r.x[idx] * rel
        cf = r.cf[idx] / np.sqrt(rel)
        keep = rex >= 1e3          # boundary-layer theory needs Re_x >> 1
        rex, cf = rex[keep], cf[keep]
        ax.loglog(rex, cf, ls="none", marker="o", ms=6, mfc="none", mec=P.INK, mew=1.1,
                  label="solver numérico ($Re_L$ = 10⁴, 10⁵, 10⁶)" if k == 0 else None)
    ax.set_xlabel(r"$Re_x = U_\infty x/\nu$")
    ax.set_ylabel(r"$c_f = 2\tau_w / \rho U_\infty^2$")
    ax.set_title("Coeficiente de atrito local na placa plana")
    ax.grid(True, which="both", lw=0.4)
    ax.set_xlim(1e3, 1e8)
    ax.legend(loc="upper right", fontsize=8.8)
    save(fig, "07_cf_reynolds.png")


# ----------------------------------------------------------------------
# 08 - animation: marching towards separation
# ----------------------------------------------------------------------
def fig08():
    from matplotlib.animation import FuncAnimation, PillowWriter

    r = howarth()
    frames = list(np.linspace(0, len(r.x) - 1, 70).astype(int)) + [len(r.x) - 1] * 12
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=90,
                               gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.25})
    X, Yg = np.meshgrid(r.x, r.y, indexing="ij")
    a.set_xlim(0, 0.13)
    a.set_ylim(0, 4.5)
    a.grid(False)
    a.set_xlabel(r"$x/L$")
    a.set_ylabel(r"$y\sqrt{Re_L}/L$")
    a.set_title("Marcha de Crank–Nicolson em x")
    a.axvline(X_SEP_HOWARTH, color=P.RED, lw=1.2, ls="--")
    (prof,) = b.plot([], [], color=P.ORANGE, lw=2)
    (ref,) = b.plot(r.u[0] / r.ue[0], r.y, color=P.INK_3, lw=1, ls=":")
    b.set_xlim(0, 1.05)
    b.set_ylim(0, 4.5)
    b.set_xlabel(r"$u/U_e$")
    b.set_title("Perfil local")
    txt = b.text(0.05, 4.0, "", fontsize=9, color=P.INK_2)
    state = {"cf": None}

    def draw(i):
        if state["cf"] is not None:
            state["cf"].remove()
        n = max(i, 2)
        state["cf"] = a.contourf(X[:n], Yg[:n], np.clip(r.u / r.ue[:, None], 0, 1)[:n],
                                 levels=np.linspace(0, 1, 21), cmap=P.SEQUENTIAL)
        prof.set_data(r.u[i] / r.ue[i], r.y)
        txt.set_text(f"x/L = {r.x[i]:.4f}\n" + r"$\tau_w$ = " + f"{r.tau_wall[i]:.3f}")
        return prof, txt

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = OUT / "08_howarth_marcha.gif"
    anim.save(path, writer=PillowWriter(fps=14))
    plt.close(fig)
    print(f"  -> {path.relative_to(OUT.parent)}")


FIGURES = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08}

if __name__ == "__main__":
    which = [int(a) for a in sys.argv[1:]] or list(FIGURES)
    for k in which:
        t0 = time.perf_counter()
        print(f"Figura {k:02d} ...")
        FIGURES[k]()
        print(f"  ({time.perf_counter() - t0:.1f} s)")
