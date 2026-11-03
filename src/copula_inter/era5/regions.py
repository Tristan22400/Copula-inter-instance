"""regions.py — named lat/lon boxes and reusable real-mode sweep-config
profiles, single-sourced here instead of the REGIONS dicts / CONFIGS lists
duplicated across every plots/*.py sweep script. Synthetic-mode's analogous
profiles (no "region" axis -- a kernel-regime axis instead) live in
constants.SYNTHETIC_SWEEP_PROFILES."""

from __future__ import annotations

# name -> (lat_bounds, lon_bounds), both (south, north) / (west, east) degrees.
REGIONS = {
    "western_europe": ((35.0, 60.0), (0.0, 30.0)),
    "north_atlantic_ocean": ((40.0, 55.0), (-40.0, -20.0)),
    "sahara_desert": ((20.0, 30.0), (0.0, 20.0)),
    "himalaya_tibetan_plateau": ((27.0, 40.0), (75.0, 95.0)),
    "amazon_basin": ((-10.0, 5.0), (-70.0, -50.0)),
}

# Each profile is a list of (config_name, region_name, grid_size) triples --
# the unit `sweep --mode real` / `diagnose --mode real` iterate over.
SWEEP_PROFILES = {
    # 3 grid resolutions over Western Europe + the 4 other regions at grid
    # 24, n_context=30 -- the profile plots/run_low_context_r2.py,
    # run_checkpoint_family_comparison.py, and
    # run_diagnostic_plots_per_checkpoint.py each hardcoded independently.
    "low_context_7config": [
        ("grid_08x08_western_europe", "western_europe", 8),
        ("grid_16x16_western_europe", "western_europe", 16),
        ("grid_24x24_western_europe", "western_europe", 24),
        ("north_atlantic_ocean", "north_atlantic_ocean", 24),
        ("sahara_desert", "sahara_desert", 24),
        ("himalaya_tibetan_plateau", "himalaya_tibetan_plateau", 24),
        ("amazon_basin", "amazon_basin", 24),
    ],
    # Phase 1's resolution sweep: Western Europe only, 6 grid sizes.
    "grid_resolution": [
        ("grid_08x08_western_europe", "western_europe", 8),
        ("grid_12x12_western_europe", "western_europe", 12),
        ("grid_16x16_western_europe", "western_europe", 16),
        ("grid_20x20_western_europe", "western_europe", 20),
        ("grid_24x24_western_europe", "western_europe", 24),
        ("grid_28x28_western_europe", "western_europe", 28),
    ],
    # Phase 2's region sweep: all 5 regions at one fixed grid size.
    "region_impact": [(name, name, 20) for name in REGIONS],
}
