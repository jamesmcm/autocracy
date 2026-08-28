# Cross-country 20-term campaigns: chronos-2 vs oracle vs no-op

Method: for each of the six missions (UK plus the five new countries) the
`campaign_trace` harness ran **20 electoral terms** (16-turn terms for
uk/germany/usa, 20-turn for australia/canada/france), saving the full state
every turn.  Three agents were compared:

| agent | description |
|---|---|
| chronos-2 | `autogluon/chronos-2` forecaster, 4 seeds (UK: 10 seeds) |
| oracle | simulator beam search (`ElectionOracleAgent`), 1 run |
| no-op | `PassiveAgent` that never spends political capital, 1 run |

Full traces: `reports/campaigns/{chronos,oracle,noop}/<country>/<seed>/`.
Plots: `reports/plots/campaign_20terms_noop_oracle_by_turn.png` (per-turn
poll rate with election boundaries and the 0.5 win barrier) and
`campaign_20terms_chronos2_vs_oracle_by_turn.png`.

## Headline results

| country | term | chronos wins | oracle wins | no-op wins | chronos poll | oracle poll | no-op poll | chronos crises/term |
|---|---|---|---|---|---|---|---|---|
| uk | 16 | 106/200 (53%) | 20/20 | 0/20 | 0.500 | 0.964 | 0.072 | 19.6 |
| australia | 20 | 50/80 (63%) | 20/20 | 0/20 | 0.552 | 0.954 | 0.022 | 12.0 |
| canada | 20 | 51/80 (64%) | 20/20 | 0/20 | 0.522 | 0.963 | 0.038 | 15.8 |
| france | 20 | 51/80 (64%) | 20/20 | **20/20** | 0.593 | 0.994 | **0.845** | 14.0 |
| germany | 16 | 70/80 (88%) | 20/20 | 0/20 | 0.680 | 0.965 | 0.432 | 10.5 |
| usa | 16 | 77/80 (96%) | 20/20 | 0/20 | 0.748 | 0.965 | 0.429 | 12.2 |

The single most striking row is **france: the no-op baseline wins all 20
terms**.  France's starting conditions are so strong that a government that
does nothing holds the country comfortably (mean poll 0.845, no debt crises).
The chronos agent, by actively intervening, turns that into a 64% win rate,
a 0.593 mean poll, and a debt spiral.

## Actions chosen: chronos vs oracle

Aggregating the recorded actions (`experiments/analyze_campaigns.py`):

| country | chronos actions (4 seeds) | oracle actions (1 run) | top chronos moves | top oracle moves |
|---|---|---|---|---|
| uk | 6225 | 347 | AlcoholTax ↓, LuxuryGoodsTax, Monorail, OrganDonation, PrivatePrisons | AgricultureSubsidies, CCTVCameras, TaxShelters, AlcoholTax ↓, ArtsSubsidies |
| australia | 3037 | 147 | FoodStandards ↓, Tasers ↓, JunkFoodTax, LuxuryGoodsTax, TaxShelters ↓ | MilitarySpending, OrganicSubsidy, EnterpriseInvestmentScheme, HealthTaxCredits |
| canada | 3132 | ~150 | PrivatePrisons ↓, FoodStandards ↓, ImportTarrifs, IncomeTax ↓, HealthcareVouchers ↓ | (targeted progressive/benefit mix) |
| france | 2847 | 616 | PrivatePrisons ↓, CarEmmissionsLimits ↓, TechnologyColleges ↓, GraduateTax ↓, PhoneTapping ↓ | TaxShelters, GraduateTax, StateHousing, FuelEfficiency, AgricultureSubsidies |
| germany | 2507 | 520 | ArtsSubsidies ↓, Monorail, PrivatePrisons ↓, RecreationalDrugsTax ↓, GatedCommunities | TaxShelters, MansionTax, OrganicSubsidy, AirlineTax, MaternityLeave |
| usa | 2536 | 494 | PrivatePrisons ↓, PoliceDrones ↓, MaternityLeave ↓, Monorail, Recycling | CCTVCameras, TaxShelters, TelecommutingInitiative, RailSubsidies, LuxuryGoodsTax |

Two structural differences stand out:

1. **Action volume and churn.**  Chronos fires thousands of moves and flips
   policies repeatedly — `PrivatePrisons` is nudged up *and* down dozens of
   times in the same life (e.g. usa: up 23 / down 60; canada: up 13 / down 74).
   The oracle makes an order of magnitude fewer, more deliberate moves and
   mostly in one direction.  The high churn indicates the forecast-based
   decisions are reacting to short-term poll wobble instead of committing to
   a durable policy position.

2. **Fiscal behaviour.**  Chronos cuts taxes/benefits heavily (`AlcoholTax` ↓,
   `FoodStandards` ↓, `IncomeTax` ↓, `MaternityLeave` ↓) while still ending
   with billions in debt; the no-op ends in surplus everywhere.  The oracle
   spends the opposite way — raises benefits and progressive taxes — and its
   debt goes even higher, yet its poll stays pinned at ~1.0 because the
   spending is well targeted at approval.

## Common problem: spending into DebtCrisis

The oracle lands every country in DebtCrisis every term (20/20) yet still
wins with ~2000-vote margins — the crisis alone is survivable if approval
stays high.  The chronos agent lands in DebtCrisis most terms too (10.5-19.6
of 20), but unlike the oracle its poll **collapses** once the crisis bites.
The failure mode is therefore not "entering a crisis" but "entering a crisis
with weak, churned approval so the crisis is never recovered from".

The UK is the worst (19.6/20 crises) because it starts already deep in debt;
germany/usa do best (10.5-12.2 crises) and win the most terms.

## France deep-dive

France is the clearest demonstration that the long-term problem is the
agent's own policy, not the country.  No-op France: 20/20 wins, 0.845 poll,
surplus.  Chronos France: 64% wins, 0.593 poll, ~44B ending debt.

All four chronos seeds follow the same shape:

```
seed 20260813: margins [1319 1502 1508 1397 1536 1282 1143 -27 ... -1533]
crisis mask:   .......CCCCCCCCCCCCC
seed 20260814: margins [1460 1451 1425 1726 1145 993 1004 809 798 924 534
                -545 -687 -779 -941 -879 -1868 -1996 -1998 -1997]
crisis mask:   ....CCCCCCCCCCCCCCCC
```

Every seed wins the first several terms comfortably (margins 800-1700), then
enters DebtCrisis around term 6-11 and thereafter bleeds margins to deep
losses (-1500 to -1998).  The debt trace for seed 20260814:

| phase | turns | debt | poll |
|---|---|---|---|
| early | 0-100 | ~0 (surplus early) | 0.80-0.96 |
| mid | 100-200 | 6.7M -> 247M | 0.72-0.80 |
| crisis | 200-240 | 458M -> 840M | 0.56 -> 0.28 |
| collapse | 240-400 | 2.8B -> 9.3B -> 56B | 0.24 -> 0.00 |

The crisis onset coincides exactly with the poll collapse and the losses
compound exponentially via debt interest.  Because France *starts* strong,
the agent is spending from a winning position and the debt spiral converts
that advantage into a terminal deficit — the opposite of what no-op does.

## Conclusions

- **The oracle is a spending ceiling, not a fiscal reference**: it maxes
  polls by spending into crisis, so matching its actions is not the goal.
- **The dominant common weakness is action churn + fiscal drift**: chronos
  reverses policies constantly and lets debt compound, and the crisis then
  locks in the damage.  No-op shows that holding policy steady is strictly
  better than churning in france (and broadly cost-free elsewhere until the
  starting state forces action).
- **France is the canary**: a country that is winnable by doing nothing is
  being lost by the agent's own interventions, so any fix that reduces
  counterproductive churn or adds a fiscal/budget guard should show up first
  and most clearly in the France curve.

## Reproducing

```bash
uv run --extra chronos python experiments/campaign_trace.py --mode noop --country france --elections 20
uv run --extra chronos python experiments/campaign_trace.py --mode chronos --country france --model autogluon/chronos-2 --seeds 4 --elections 20
uv run --extra chronos python experiments/campaign_trace.py --mode oracle --country france --elections 20
uv run --with matplotlib python experiments/plot_campaign_compare.py
uv run python experiments/analyze_campaigns.py
```