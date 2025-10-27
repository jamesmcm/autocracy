from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from autocracy.models import PolicyAction
from autocracy import simulator
from autocracy.agent import PassiveAgent
from autocracy.savegame import (
    compare_state_to_savegame,
    load_state_from_savegame,
    parse_savegame,
)

app = typer.Typer(help="Lightweight Democracy 3 simulator harness.")
console = Console()
DEFAULT_METRICS = [
    "GDP",
    "Health",
    "Education",
    "CrimeRate",
    "Unemployment",
]


def _prepare_data(gamedata: Optional[Path]) -> simulator.SimulationData:
    return simulator.load_simulation_data(str(gamedata)) if gamedata else simulator.load_simulation_data()


def _print_metrics(state, metrics: List[str]):
    table = Table(title=f"Turn {state.turn} snapshot", show_edge=False, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name in metrics:
        value = state.values.get(name)
        if value is None:
            continue
        table.add_row(name, f"{value:.3f}")
    console.print(table)
    balance = state.total_income - state.total_expenditure
    console.print(
        "Political capital: "
        f"{state.political_capital:.2f} | Income: {state.total_income:,.0f} | "
        f"Expenditure: {state.total_expenditure:,.0f} | Balance: {balance:,.0f}"
    )


def _print_full_state(state, data: simulator.SimulationData):
    node_table = Table(title="Full DAG state", show_edge=False, header_style="bold cyan")
    node_table.add_column("Node")
    node_table.add_column("Value", justify="right")
    node_table.add_column("Min", justify="right")
    node_table.add_column("Max", justify="right")
    for name in sorted(data.nodes.keys()):
        node = data.nodes[name]
        value = state.values.get(name, node.default)
        node_table.add_row(name, f"{value:.3f}", f"{node.minimum:.2f}", f"{node.maximum:.2f}")
    console.print(node_table)

    policy_table = Table(title="Policy levels", show_edge=False, header_style="bold cyan")
    policy_table.add_column("Policy")
    policy_table.add_column("Level", justify="right")
    policy_table.add_column("Status")
    policy_table.add_column("Introduce", justify="right")
    policy_table.add_column("Cancel", justify="right")
    policy_table.add_column("Raise/Lower", justify="right")
    policy_table.add_column("Delay", justify="right")
    policy_table.add_column("Cost", justify="right")
    policy_table.add_column("Income", justify="right")
    for name in sorted(data.policies.keys()):
        policy = data.policies[name]
        level = state.policies.get(name, 0.0)
        status = "active" if level > simulator.EPSILON else "not implemented"
        policy_table.add_row(
            name,
            f"{level:.2f}",
            status,
            f"{policy.introduce_cost:.0f}",
            f"{policy.cancel_cost:.0f}",
            f"{policy.raise_cost:.0f}/{policy.lower_cost:.0f}",
            f"{policy.implementation_time:.0f}",
            f"{state.policy_costs.get(name, 0.0):,.0f}",
            f"{state.policy_incomes.get(name, 0.0):,.0f}",
        )
    console.print(policy_table)


def _resolve_graph_node(graph, name: str) -> Optional[str]:
    candidate = name.strip()
    if candidate in graph:
        return candidate
    lowered = candidate.lower()
    for node in graph.nodes:
        if node.lower() == lowered:
            return node
        data = graph.nodes[node].get("data")
        display = getattr(data, "display_name", "")
        if display and display.lower() == lowered:
            return node
    return None


def _format_inertia(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _print_effects_table(title: str, effects, is_input: bool):
    if not effects:
        console.print(f"{title}: none")
        return
    table = Table(title=title, show_edge=False, header_style="bold cyan")
    table.add_column("Source" if is_input else "Target")
    table.add_column("Expression")
    table.add_column("Inertia", justify="right")
    for effect in effects:
        endpoint = effect.source if is_input else effect.target
        table.add_row(endpoint, effect.expression, _format_inertia(effect.inertia))
    console.print(table)


def _print_comparison(title: str, diffs, missing):
    if not diffs and not missing:
        console.print(f"[green]{title}: no differences within tolerance[/green]")
        return
    if diffs:
        table = Table(title=f"{title} differences", show_edge=False, header_style="bold yellow")
        table.add_column("Name")
        table.add_column("Simulator", justify="right")
        table.add_column("Savegame", justify="right")
        table.add_column("Delta", justify="right")
        for diff in diffs:
            table.add_row(
                diff.name,
                f"{diff.simulator:.3f}",
                f"{diff.savegame:.3f}",
                f"{diff.delta:+.3f}",
            )
        console.print(table)
    if missing:
        console.print(f"[red]{title}: missing entries[/red] {', '.join(sorted(missing))}")


def _parse_policy_changes(changes: List[str]) -> List[PolicyAction]:
    actions: List[PolicyAction] = []
    for item in changes:
        if ":" not in item:
            raise typer.BadParameter("Policy changes must be formatted as Name:delta (e.g. IncomeTax:-0.05)")
        name, delta = item.split(":", 1)
        actions.append(PolicyAction(policy_name=name.strip(), delta=float(delta)))
    return actions


@app.command()
def describe(
    country: str = typer.Option("uk", "--country", "-c", help="Country mission to load."),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    metrics: List[str] = typer.Option([], "--metric", "-m", help="Specific metrics to display."),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show the entire DAG state and policy list.",
    ),
):
    """Show the initial conditions for a country."""

    data = _prepare_data(gamedata)
    state, graph = simulator.get_initial_state(country, str(gamedata) if gamedata else None)
    console.print(f"Loaded {country.upper()} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    metric_list = metrics or DEFAULT_METRICS
    _print_metrics(state, metric_list)
    if verbose:
        _print_full_state(state, data)


@app.command()
def simulate(
    country: str = typer.Option("uk", "--country", "-c", help="Country mission to load."),
    turns: int = typer.Option(4, "--turns", "-t", help="Number of turns to simulate."),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    metrics: List[str] = typer.Option([], "--metric", "-m", help="Metrics to print each turn."),
    policy: List[str] = typer.Option(
        [],
        "--policy",
        "-p",
        help="Apply a policy delta before running (format: Name:delta). Repeat for multiple changes.",
    ),
    state_in: Optional[Path] = typer.Option(
        None,
        "--state-in",
        help="Load a saved state snapshot before running turns.",
    ),
    state_out: Optional[Path] = typer.Option(
        None,
        "--state-out",
        help="Persist the final state snapshot after the run.",
    ),
):
    """Run the deterministic DAG update loop for the requested number of turns."""

    data = _prepare_data(gamedata)
    if state_in:
        state = simulator.load_state(state_in)
        graph = simulator.build_country_graph(state.country, str(gamedata) if gamedata else None)
    else:
        state, graph = simulator.get_initial_state(
            country, str(gamedata) if gamedata else None
        )
    metric_list = metrics or DEFAULT_METRICS
    if policy:
        actions = _parse_policy_changes(policy)
        state = simulator.apply_actions(state, actions, data=data)
        console.print("[green]Applied policy changes[/green]")
    _print_metrics(state, metric_list)
    for _ in range(turns):
        state = simulator.process_end_of_turn(state, graph, data=data)
        _print_metrics(state, metric_list)
    if state_out:
        simulator.save_state(state, state_out)
        console.print(f"Saved state to {state_out}")


@app.command()
def node(
    name: str = typer.Argument(..., help="Simulation node or policy to inspect."),
    country: str = typer.Option("uk", "--country", "-c", help="Country mission to load."),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
):
    """Show all simulator inputs and outputs for a node."""

    data = _prepare_data(gamedata)
    data_root = str(gamedata) if gamedata else None
    graph = simulator.build_country_graph(country, data_root)
    resolved = _resolve_graph_node(graph, name)
    if resolved is None:
        raise typer.BadParameter(f"Node '{name}' not found in the {country.upper()} graph.")
    metadata = graph.nodes[resolved].get("data")
    display = getattr(metadata, "display_name", resolved)
    category = getattr(metadata, "category", getattr(metadata, "department", ""))
    heading = f"[bold]{display}[/bold] ({resolved})"
    if category:
        heading = f"{heading} - {category}"
    console.print(heading)
    description = getattr(metadata, "description", "")
    if description:
        console.print(description)
    inputs, outputs = simulator.collect_node_effects(resolved, graph, data=data)
    _print_effects_table("Inputs", inputs, is_input=True)
    _print_effects_table("Outputs", outputs, is_input=False)


@app.command()
def actions(
    country: str = typer.Option("uk", "--country", "-c", help="Country mission to load."),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    policy: List[str] = typer.Option(
        [],
        "--policy",
        "-p",
        help="Apply a policy delta before listing options (format: Name:delta).",
    ),
    state_in: Optional[Path] = typer.Option(
        None,
        "--state-in",
        help="Load a saved state snapshot before enumerating options.",
    ),
):
    """List feasible single-step policy moves from the current state."""

    data = _prepare_data(gamedata)
    if state_in:
        state = simulator.load_state(state_in)
    else:
        state, _ = simulator.get_initial_state(
            country, str(gamedata) if gamedata else None
        )
    if policy:
        actions = _parse_policy_changes(policy)
        state = simulator.apply_actions(state, actions, data=data)
        console.print("[green]Applied policy changes[/green]")
    options = simulator.list_available_actions(state, data=data)
    if not options:
        console.print("No actions available with the current political capital.")
        return
    table = Table(title=f"Available actions for {country.upper()}", show_edge=False, header_style="bold cyan")
    table.add_column("Policy")
    table.add_column("Action")
    table.add_column("Delta", justify="right")
    table.add_column("Result", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Delay", justify="right")
    for option in options:
        table.add_row(
            option.policy_name,
            option.action_type,
            f"{option.delta:+.2f}",
            f"{option.resulting_level:.2f}",
            f"{option.cost:.0f}",
            f"{option.implementation_time:.0f}",
        )
    console.print(table)


@app.command()
def agent(
    country: str = typer.Option("uk", "--country", "-c", help="Country mission to load."),
    turns: int = typer.Option(
        4, "--turns", "-t", help="Number of decision loops to execute."
    ),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    state_in: Optional[Path] = typer.Option(
        None,
        "--state-in",
        help="Start from a previously saved state snapshot.",
    ),
    state_out: Optional[Path] = typer.Option(
        None,
        "--state-out",
        help="Persist the final state snapshot for later comparison.",
    ),
    metrics: List[str] = typer.Option([], "--metric", "-m", help="Metrics to show after each loop."),
):
    """Run the baseline agent loop that alternates between choosing actions and ending turns."""

    data_root = str(gamedata) if gamedata else None
    if state_in:
        state = simulator.load_state(state_in)
        agent = PassiveAgent(
            country=state.country,
            gamedata_root=data_root,
            state=state,
        )
    else:
        agent = PassiveAgent(country=country, gamedata_root=data_root)
    metric_list = metrics or DEFAULT_METRICS
    console.print(
        f"Starting agent loop for {agent.state.country.upper()} with passive strategy"
    )
    for _ in range(turns):
        agent.step()
        _print_metrics(agent.state, metric_list)
    if state_out:
        agent.save_state(state_out)
        console.print(f"Saved state to {state_out}")


@app.command()
def load_save(
    savefile: Path = typer.Argument(..., exists=True, readable=True, help="Path to a Democracy 3 savegame XML."),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    metrics: List[str] = typer.Option([], "--metric", "-m", help="Metrics to display."),
):
    """Load a Democracy 3 savegame and display the simulator state derived from it."""

    save = parse_savegame(savefile)
    data_root = str(gamedata) if gamedata else None
    state, _ = load_state_from_savegame(savefile, data_root)
    console.print(
        f"Loaded save '{savefile}' for {save.country.upper()} (turn {save.turn}) with {len(save.simvalues)} nodes."
    )
    metric_list = metrics or DEFAULT_METRICS
    _print_metrics(state, metric_list)


@app.command()
def compare_save(
    savefile: Path = typer.Argument(..., exists=True, readable=True, help="Path to a Democracy 3 savegame XML."),
    state_in: Optional[Path] = typer.Option(
        None,
        "--state-in",
        help="Simulator JSON snapshot to compare. If omitted, uses state derived from the savefile.",
    ),
    gamedata: Optional[Path] = typer.Option(None, "--gamedata", help="Override gamedata/data path."),
    tolerance: float = typer.Option(
        1e-3,
        "--tolerance",
        "-t",
        help="Maximum absolute difference treated as equal.",
    ),
):
    """Compare a simulator state against a Democracy 3 save to inspect divergences."""

    save = parse_savegame(savefile)
    data_root = str(gamedata) if gamedata else None
    if state_in:
        state = simulator.load_state(state_in)
        if state.country.lower() != save.country.lower():
            raise typer.BadParameter(
                f"State country {state.country} does not match save country {save.country}."
            )
    else:
        state, _ = load_state_from_savegame(savefile, data_root)
    comparison = compare_state_to_savegame(state, save, tolerance=tolerance)
    if not comparison.has_differences():
        console.print("[green]Simulator state matches savegame within tolerance.[/green]")
        return
    _print_comparison("Node values", comparison.value_diffs, comparison.missing_values)
    _print_comparison("Policies", comparison.policy_diffs, comparison.missing_policies)
    _print_comparison("Policy costs", comparison.cost_diffs, comparison.missing_costs)
    _print_comparison("Policy incomes", comparison.income_diffs, comparison.missing_incomes)
    _print_comparison("Budget totals", comparison.budget_diffs, [])


if __name__ == "__main__":
    app()
