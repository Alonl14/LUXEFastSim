"""Canonical detector-disc region boundaries.

Single source of truth shared by data splitting and evaluation. Boundaries
are half-open so the three masks partition the plane exactly once. They match
the intent of the cuts historically used in utils.check_run (yy threshold 520,
xx thresholds 500 and -1700), replacing the inconsistent cuts in MakePDGPlots.
"""

# Training order: outer regions first, then inner (preserved from FastSimCluster).
REGIONS = ("outer1", "outer2", "inner")

DISC_RX_MAX = 4000.0
TIME_MAX = 1e6


def region_mask(df, region):
    xx = df[" xx"]
    yy = df[" yy"]
    if region == "outer1":
        return (xx >= 500) | (yy >= 520)
    if region == "outer2":
        return (xx < -1700) & (yy < 520)
    if region == "inner":
        return (xx >= -1700) & (xx < 500) & (yy < 520)
    raise ValueError(f"Unknown region: {region!r}; expected one of {REGIONS}")
