"""Laminar boundary-layer theory and numerics in the 2D plane.

Modules
-------
similarity  Blasius and Falkner–Skan similarity solutions.
marching    Crank–Nicolson space-marching solver for the Prandtl equations.
integral    Von Kármán momentum integral: Pohlhausen profiles and Thwaites.
turbulent   Empirical turbulent flat-plate correlations (context only).
"""

from .integral import PROFILES, thwaites
from .marching import MarchingResult, marching_grid, solve, wall_grid
from .similarity import blasius, blasius_constants, falkner_skan

__all__ = [
    "PROFILES",
    "MarchingResult",
    "blasius",
    "blasius_constants",
    "falkner_skan",
    "marching_grid",
    "solve",
    "thwaites",
    "wall_grid",
]

__version__ = "1.0.0"
