## Democracy 3 Simulator (WIP)

This repository contains a lightweight simulator for Democracy 3 that operates directly on the original `data` assets.

Place the Democracy 3 `data` folder inside the `gamedata` directory - so
it should be `./gamedata/data/simconfig.txt` for example.

Note that the game data is encoded with latin-1 not UTF-8.

The aim is to create an environment for online learning AI agents - i.e. can they learn without failure without knowing the underlying DAG and equations?

Primarily vibe-coded with OpenAI Codex.

### Running the CLI

Use [uv](https://github.com/astral-sh/uv) (already configured for this project):

```bash
# Inspect the initial UK setup
uv run main.py describe --country uk

# Dump the entire DAG state and policy roster
uv run main.py describe --country uk --verbose

# Apply a policy tweak, advance one turn, and persist the resulting state
uv run main.py simulate --country uk --turns 1 -p IncomeTax:-0.05 --state-out snapshot.json

# Resume from a saved snapshot with the passive agent loop
uv run main.py agent --state-in snapshot.json --turns 3

# Load a Democracy 3 save directly
uv run main.py load-save gamedata/saves/uk0.xml

# Compare a saved simulator snapshot against a savegame
uv run main.py compare-save gamedata/saves/uk0.xml --state-in snapshot.json

# Inspect the causes/effects for a specific node (e.g. Health)
uv run main.py node Health

# Watch the budget summary each turn
uv run main.py simulate --country uk --turns 2
```

Options:

- `--metric` / `-m`: limit the metrics printed each turn (defaults to GDP, Health, Education, CrimeRate, Unemployment).
- `--policy` / `-p`: repeatable flag to apply policy slider deltas before the run (`Name:delta` format).
- `--gamedata`: point the simulator at an alternate `gamedata/data` directory if you want to swap in a different set of Democracy 3 assets.
- `--verbose`: available on `describe` to output the full DAG and every policy with costs/delays plus the live cost/income columns for each slider.
- Every metrics print now ends with political capital, total income, total expenditure, and the net balance so you can track top-level finances turn by turn.
- New command `node <Name>` prints every inbound/outbound effect so you can debug edges such as those shown in `health.png`.
- `--state-in`/`--state-out`: load or persist JSON snapshots produced by the simulator for A/B comparisons against the real game.

### Architecture

- `autocracy/data_loader.py` parses the CSV/INI files under `gamedata/data/`.
- `autocracy/simulator.py` builds the DAG, exposes helpers to obtain an initial state, process turns, and apply actions, and leaves `dilemmas/attacks/events` as explicit stubs for future work.
- `autocracy/agent.py` defines a turn-by-turn agent scaffold (`BaseAgent`/`PassiveAgent`) that future learning agents can inherit from.
- `autocracy/savegame.py` parses Democracy 3 XML save files, lets you initialize simulator state from a save, and compares simulator output against real save snapshots.
- `main.py` provides a Typer-based CLI that exercises the simulator functions, including state save/load flows, and prints human-friendly tables with Rich.
- `SIMULATION.md` documents the data formats, DAG processing rules, state persistence, and the public simulator APIs in more depth.

### Situation + inertia handling

- `situations.csv` is now parsed alongside the rest of the simulation assets, so latent situation values (Street Gangs, Debt Crisis, etc.) are evaluated every turn using their trigger thresholds and per-link inertia.
- The simulator keeps an effect memory for every edge/situation link and applies the Democracy 3-style inertia rule (`value += (target - value) / inertia`) so delayed inputs (e.g., Community Policing → Street Gangs) and outputs (e.g., Street Gangs → CrimeRate) ramp up/down across multiple turns instead of jumping immediately.
- `SimulationState` snapshots now include `situations` (latent values) and `active_situations` to make it easy to inspect or persist which modifiers are currently impacting the DAG.
- When available, the simulator seeds its initial node + policy values from the vanilla save in `gamedata/saves/<country>0.xml`, guaranteeing that running `get_initial_state("uk")` starts from the same GDP/Crime baseline the real game ships with.
- If the matching `...1.xml` save is present, the simulator derives per-node “response factors” that scale each turn’s aggregate delta so that a no-op first turn reproduces the corresponding savegame snapshot. These factors are persisted with the state so downstream comparisons stay anchored to the observed Democracy 3 behaviour.

### Savegame integration

```python
from autocracy.savegame import parse_savegame, load_state_from_savegame, compare_state_to_savegame
from autocracy.simulator import state_to_dict

save = parse_savegame("gamedata/saves/uk0.xml")
state, graph = load_state_from_savegame("gamedata/saves/uk0.xml")
comparison = compare_state_to_savegame(state, save)
print("Differences?", comparison.has_differences())
```

Use this to validate the simulator against real Democracy 3 save files or to bootstrap the simulator from a known in-game state.

# TODOs

- There are still some issues with simulation.csv latent nodes and
  monetary costs:


uv run main.py compare-save gamedata/saves/uk1.xml --state-in ukout1.json
              Node values differences
 Name               ┃ Simulator ┃ Savegame ┃  Delta
━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━
 Health             │     0.103 │    0.602 │ -0.499
 CurrencyStrength   │     0.000 │    0.233 │ -0.233
 Education          │     0.299 │    0.476 │ -0.177
 Equality           │     0.497 │    0.607 │ -0.110
 PrivatePensions    │     0.208 │    0.120 │ +0.088
 Immigration        │     0.321 │    0.385 │ -0.065
 _Terrorism         │     0.245 │    0.189 │ +0.056
 PrivateHealthcare  │     0.217 │    0.166 │ +0.051
 ViolentCrimeRate   │     0.279 │    0.238 │ +0.041
 PrivateHousing     │     0.234 │    0.193 │ +0.041
 CarUsage           │     0.670 │    0.699 │ -0.029
 GDP                │     0.191 │    0.169 │ +0.022
 AlcoholConsumption │     0.646 │    0.667 │ -0.021
 RacialTension      │     0.498 │    0.496 │ +0.002
Policies: no differences within tolerance
                  Policy costs differences
 Name                  ┃ Simulator ┃  Savegame ┃      Delta
━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━
 StateHealthService    │  8856.795 │ 46481.637 │ -37624.842
 StatePensions         │  9400.002 │ 46070.188 │ -36670.186
 MilitarySpending      │  5348.916 │ 25830.666 │ -20481.750
 StateHousing          │  4700.000 │ 23035.094 │ -18335.094
 StateSchools          │  4182.144 │ 21081.043 │ -16898.899
 RailSubsidies         │  3209.000 │ 15538.858 │ -12329.858
 ChildBenefit          │  3144.000 │ 15409.007 │ -12265.007
 RoadBuilding          │  2600.000 │ 12589.914 │  -9989.914
 UnemployedBenefit     │  1021.708 │  6184.898 │  -5163.190
 ScienceFunding        │  1259.500 │  6419.256 │  -5159.756
 PoliceForce           │  1270.549 │  6359.279 │  -5088.730
 IntelligenceServices  │  1250.000 │  6325.864 │  -5075.864
 AgricultureSubsidies  │  1192.000 │  6041.145 │  -4849.145
 ForeignAid            │  1116.000 │  5449.134 │  -4333.134
 Prisons               │  1010.000 │  5088.505 │  -4078.505
 CCTVCameras           │   656.000 │  3319.813 │  -2663.813
 CommunityPolicing     │   439.570 │  2224.528 │  -1784.958
 JuryTrial             │   213.500 │  1080.458 │   -866.958
 LegalAid              │   205.000 │  1037.442 │   -832.442
 CleanEnergySubsidies  │   406.767 │  1168.292 │   -761.525
 BusLanes              │   160.000 │   774.764 │   -614.764
 BorderControls        │   145.000 │   707.997 │   -562.997
 LabourLaws            │   130.000 │   658.850 │   -528.850
 Recycling             │   122.800 │   622.360 │   -499.560
 FoodStandards         │    47.800 │   243.621 │   -195.821
 PollutionControls     │    17.000 │    86.157 │    -69.157
 RaceDiscriminationAct │     8.200 │    41.498 │    -33.298
 HandgunLaws           │     2.000 │    10.121 │     -8.121
              Policy incomes differences
 Name            ┃ Simulator ┃   Savegame ┃      Delta
━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━
 IncomeTax       │ 85402.632 │ 133753.484 │ -48350.852
 PetrolTax       │  4770.665 │  20445.178 │ -15674.513
 PropertyTax     │  3424.528 │  13534.515 │ -10109.987
 SalesTax        │ 34637.632 │  25877.127 │  +8760.505
 CorporationTax  │ 10804.508 │  16921.500 │  -6116.992
 AlcoholTax      │  1299.430 │   5518.602 │  -4219.172
 InheritanceTax  │   827.080 │   3893.609 │  -3066.529
 TobaccoTax      │   745.290 │   2954.885 │  -2209.595
 CapitalGainsTax │  3192.841 │   5000.474 │  -1807.633
                 Budget totals differences
 Name              ┃  Simulator ┃   Savegame ┃       Delta
━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━
 Total Expenditure │  52114.252 │ 259880.389 │ -207766.137
 Total Income      │ 145104.607 │ 227899.374 │  -82794.767

