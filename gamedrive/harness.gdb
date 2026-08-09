set pagination off
set confirm off
set print thread-events off
set unwind-on-signal on
handle SIGSEGV stop print nopass
handle SIGFPE stop print nopass

break mainLoop
run -silent

printf "== STOPPED at mainLoop: the simulation is reachable without the GUI resize crash ==\n"

# Access the live simulation singleton.
set $sim = (void*)SIM_GetSimulation()
printf "SIM_GetSimulation() = %p\n", $sim

printf "== entry points available ==\n"
printf "  SIM_Simulation::NextTurn         = 0x60e120\n"
printf "  SIM_Simulation::GetNeuronByName  = 0x60c140\n"
printf "  SIM_Simulation::Initialise       = 0x60f560\n"
printf "  SIM_Policy::ForceSlider          = 0x5f5ba0\n"
printf "  SIM_LoadGame::OpenSavedFile      = 0x5d4a90\n"
printf "  SIM_LoadGame::LoadGameData       = 0x5dcba0\n"

printf "== attempting SIM_Simulation::NextTurn() (expected to fault without a country) ==\n"
call ((void(*)(void*))0x60e120)($sim)
printf "NextTurn returned cleanly\n"

printf "== NOTE: to advance a real game, load a country first (see gamedrive/README.md) ==\n"
quit
