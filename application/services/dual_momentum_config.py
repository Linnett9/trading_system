from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def active_dual_momentum_config(
    config: dict[str, Any],
    use_frozen_champion: bool = True,
) -> dict[str, Any]:
    """Return the runtime dual-momentum config, applying frozen champion overrides."""
    base_config = deepcopy(config["research"].get("dual_momentum", {}))

    if not use_frozen_champion:
        return base_config

    if not base_config.get("champion_mutation_guard_enabled", False):
        return base_config

    champion_path = base_config.get("champion_config_path")

    if not champion_path:
        return base_config

    champion_file = Path(champion_path)

    if not champion_file.exists():
        return base_config

    payload = yaml.safe_load(champion_file.read_text(encoding="utf-8")) or {}

    if payload.get("do_not_mutate") and not payload.get("frozen"):
        raise ValueError(
            f"Champion config {champion_file} is marked do_not_mutate but is not frozen."
        )

    merged_config = deepcopy(base_config)
    merged_config.update(payload.get("overrides", {}))
    merged_config["champion_id"] = payload.get(
        "champion_id",
        base_config.get("champion_id"),
    )
    merged_config["champion_source_config_name"] = payload.get(
        "source_config_name",
    )
    merged_config["champion_config_path"] = str(champion_file)
    merged_config["experiment_name"] = payload.get(
        "source_config_name",
        merged_config.get("experiment_name"),
    )
    return merged_config
