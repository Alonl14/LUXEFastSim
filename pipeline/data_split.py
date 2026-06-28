"""Split a master sample into per-region CSVs using the canonical region masks."""

import os

from pipeline.regions import REGIONS, region_mask


def split_by_region(df):
    return {region: df[region_mask(df, region)] for region in REGIONS}


def split_particle(master_df, pdg, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    df = master_df[master_df[" pdg"] == pdg]
    paths = {}
    for region, part in split_by_region(df).items():
        path = os.path.join(out_dir, f"{prefix}{pdg}_{region}.csv")
        part.to_csv(path, index=False)
        paths[region] = path
    return paths
