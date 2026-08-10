"""
High-level helpers exposed for consumers of the simulator package.
"""

from .simulator import (
    apply_actions,
    apply_native_manager_roster,
    apply_native_effect_histories,
    apply_native_voter_runtime,
    build_country_graph,
    get_initial_state,
    list_available_actions,
    load_simulation_data,
    load_state,
    process_end_of_turn,
    recompute_effects,
    resolve_election,
    save_state,
    state_from_dict,
    state_to_dict,
)
from .savegame import (
    compare_state_to_savegame,
    load_state_from_savegame,
    parse_savegame,
    state_from_savegame,
)

__all__ = [
    "apply_actions",
    "apply_native_manager_roster",
    "apply_native_effect_histories",
    "apply_native_voter_runtime",
    "build_country_graph",
    "get_initial_state",
    "list_available_actions",
    "load_simulation_data",
    "load_state",
    "process_end_of_turn",
    "recompute_effects",
    "resolve_election",
    "save_state",
    "state_from_dict",
    "state_to_dict",
    "parse_savegame",
    "state_from_savegame",
    "load_state_from_savegame",
    "compare_state_to_savegame",
]
