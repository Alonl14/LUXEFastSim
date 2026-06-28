"""Materialize per-region training configs from a base template + per-pdg override."""

import json
import os

from pipeline.regions import REGIONS


def load_base(path):
    with open(path, "r") as fp:
        return json.load(fp)


def load_override(overrides_dir, pdg):
    path = os.path.join(overrides_dir, f"{pdg}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fp:
        return json.load(fp)


def _merged_shared(base, override):
    shared = dict(base.get("shared", {}))
    shared.update(override.get("shared", {}))
    return shared


def _features_for(base, override, region):
    by_region = override.get("featuresByRegion", {})
    if region in by_region:
        return by_region[region]
    if "defaultFeatures" in override:
        return override["defaultFeatures"]
    return base["defaultFeatures"]


def materialize_region_config(base, override, pdg, region, output_dir, num_epochs):
    shared = _merged_shared(base, override)
    data_root = override.get("dataRoot", base["dataRoot"])
    data_prefix = override.get("dataPrefix", base.get("dataPrefix", ""))

    cfg = dict(shared)
    cfg["pdg"] = pdg
    cfg["dataGroup"] = region
    cfg["outputDir"] = output_dir
    cfg["numEpochs"] = num_epochs
    cfg["data_path"] = os.path.join(data_root, f"{data_prefix}{pdg}_{region}.csv")
    cfg["features"] = _features_for(base, override, region)
    return cfg


def materialize_all(base, override, pdg, output_dir, num_epochs):
    return {
        region: materialize_region_config(base, override, pdg, region,
                                          output_dir, num_epochs)
        for region in REGIONS
    }
