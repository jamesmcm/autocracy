from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .models import PolicyAction, SimulationConfig, SimulationState
from . import simulator


class BaseAgent:
    """Minimal agent loop scaffold that can be subclassed for custom strategies."""

    def __init__(
        self,
        country: str = "uk",
        gamedata_root: Optional[str | Path] = None,
        state: Optional[SimulationState] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.gamedata_root = str(gamedata_root) if gamedata_root else None
        self.data = simulator.load_simulation_data(self.gamedata_root)
        self.config = config
        if state is None:
            self.state, self.graph = simulator.get_initial_state(
                country, self.gamedata_root
            )
        else:
            self.state = state
            self.graph = simulator.build_country_graph(
                state.country, self.gamedata_root
            )

    def available_actions(self):
        return simulator.list_available_actions(self.state, data=self.data)

    def choose_actions(self, state: SimulationState, options) -> Iterable[PolicyAction]:
        """Override to select actions before ending the turn. Default: no actions."""

        return []

    def apply_actions(self, actions: Iterable[PolicyAction]):
        if not actions:
            return
        self.state = simulator.apply_actions(self.state, actions, data=self.data)

    def end_turn(self):
        self.state = simulator.process_end_of_turn(
            self.state, self.graph, data=self.data, config=self.config
        )

    def step(self) -> SimulationState:
        actions = list(self.choose_actions(self.state, self.available_actions()))
        self.apply_actions(actions)
        self.end_turn()
        return self.state

    def load_state(self, path: str | Path) -> None:
        self.state = simulator.load_state(path)
        self.graph = simulator.build_country_graph(
            self.state.country, self.gamedata_root
        )

    def save_state(self, path: str | Path) -> None:
        simulator.save_state(self.state, path)


class PassiveAgent(BaseAgent):
    """Baseline agent that never spends political capital."""

    def choose_actions(self, state: SimulationState, options):
        return []
