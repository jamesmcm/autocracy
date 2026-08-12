"""
High-level helpers exposed for consumers of the simulator package.
"""

from .simulator import (
    apply_actions,
    apply_native_manager_roster,
    apply_native_sim_values,
    apply_native_effect_histories,
    apply_native_policy_runtime,
    apply_native_finance_runtime,
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
from .agent import BaseAgent, OracleAgent, PassiveAgent, SimulatorOracleAgent
from .oracle import (
    DEFAULT_ORACLE_WEIGHTS,
    OracleSearchResult,
    score_savegame,
    score_simulation_state,
)

__all__ = [
    "apply_actions",
    "apply_native_manager_roster",
    "apply_native_sim_values",
    "apply_native_effect_histories",
    "apply_native_policy_runtime",
    "apply_native_finance_runtime",
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
    "BaseAgent",
    "PassiveAgent",
    "OracleAgent",
    "SimulatorOracleAgent",
    "DEFAULT_ORACLE_WEIGHTS",
    "OracleSearchResult",
    "score_savegame",
    "score_simulation_state",
]
