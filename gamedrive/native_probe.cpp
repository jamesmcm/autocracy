// Version-pinned in-process probe for Democracy 3 v1.30.2.
//
// The game is an ET_EXEC binary, so these addresses are intentionally absolute
// and must be revalidated with preflight.py whenever the binary changes.  The
// library is loaded with LD_PRELOAD, but it does not call into the game from
// its ELF constructor.  gdb invokes the exported functions after mainLoop,
// when the executable's own global initialization is complete.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

constexpr std::uintptr_t kLoadGuard = 0xA3A9F8;
constexpr std::uintptr_t kLoadObject = 0xA3AA00;
constexpr std::uintptr_t kLoadConstructor = 0x5D39C0;
constexpr std::uintptr_t kLoadGame = 0x5D4CD0;

constexpr std::uintptr_t kSaveGuard = 0xA3A9F0;
constexpr std::uintptr_t kSaveObject = 0xA3B5C0;
constexpr std::uintptr_t kSaveConstructor = 0x602940;
constexpr std::uintptr_t kSaveGame = 0x608A40;

constexpr std::uintptr_t kSimulationGetter = 0x5AB1A0;
constexpr std::uintptr_t kNextTurn = 0x60E120;
constexpr std::uintptr_t kGameplayGetter = 0x5B9A80;
constexpr std::uintptr_t kGameplayNextTurn = 0x5D0F80;
constexpr std::uintptr_t kNextTurnThread = 0x5CFF30;
constexpr std::uintptr_t kGetNeuronByName = 0x60C140;
constexpr std::size_t kNeuronValueOffset = 0x38;
constexpr std::uintptr_t kLoadingCompleteFlag = 0xA15C50;

using Constructor = void (*)(void*);
using LoadGameFunction = void (*)(void*, const std::string&);
using SaveGameFunction = void (*)(void*, const std::string&);
using SimulationGetter = void* (*)();
using NextTurnFunction = void (*)(void*);
using NextTurnThreadFunction = int (*)(void*);
using GetNeuronFunction = void* (*)(void*, std::string);

bool g_load_started = false;
bool g_probe_started = false;
bool g_turn_started = false;
bool g_probe_finished = false;

template <typename Function>
Function function_at(std::uintptr_t address) {
    return reinterpret_cast<Function>(address);
}

unsigned char& guard_at(std::uintptr_t address) {
    return *reinterpret_cast<unsigned char*>(address);
}

void ensure_object(std::uintptr_t guard_address, std::uintptr_t object_address,
                   std::uintptr_t constructor_address, const char* label) {
    if (guard_at(guard_address) != 0) {
        return;
    }

    function_at<Constructor>(constructor_address)(
        reinterpret_cast<void*>(object_address));
    // The native inline getter checks the first guard byte before returning the
    // object.  The constructor is deliberately called here, outside the
    // getter, so gdb can trigger the same lazy-singleton transition safely.
    guard_at(guard_address) = 1;
    std::fprintf(stderr, "[d3probe] constructed %s at 0x%zx\n", label,
                 object_address);
}

const char* setting(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : fallback;
}

bool save_game(const char* setting_name, const char* fallback) {
    ensure_object(kSaveGuard, kSaveObject, kSaveConstructor, "SIM_SaveGame");
    const std::string name(setting(setting_name, fallback));
    std::fprintf(stderr, "[d3probe] saving native state as '%s'\n", name.c_str());
    function_at<SaveGameFunction>(kSaveGame)(
        reinterpret_cast<void*>(kSaveObject), name);
    return true;
}

bool save_game_name(const std::string& name) {
    ensure_object(kSaveGuard, kSaveObject, kSaveConstructor, "SIM_SaveGame");
    std::fprintf(stderr, "[d3probe] saving native state as '%s'\n", name.c_str());
    function_at<SaveGameFunction>(kSaveGame)(
        reinterpret_cast<void*>(kSaveObject), name);
    return true;
}

void* neuron_by_name(const char* name) {
    if (name == nullptr || *name == '\0') {
        return nullptr;
    }

    void* simulation = function_at<SimulationGetter>(kSimulationGetter)();
    const std::string neuron_name(name);
    return function_at<GetNeuronFunction>(kGetNeuronByName)(simulation,
                                                              neuron_name);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int d3_start_load() {
    if (g_load_started) {
        return 0;
    }
    g_load_started = true;

    ensure_object(kLoadGuard, kLoadObject, kLoadConstructor, "SIM_LoadGame");
    const std::string name(setting("D3_LOAD_NAME", "turn0_initial"));
    std::fprintf(stderr, "[d3probe] starting native load of '%s'\n", name.c_str());
    function_at<LoadGameFunction>(kLoadGame)(
        reinterpret_cast<void*>(kLoadObject), name);
    std::fprintf(stderr, "[d3probe] LoadGame returned; worker continues asynchronously\n");
    return 0;
}

extern "C" __attribute__((visibility("default"))) int d3_next_turn() {
    void* simulation = function_at<SimulationGetter>(kSimulationGetter)();
    std::fprintf(stderr, "[d3probe] calling native SIM_Simulation::NextTurn()\n");
    function_at<NextTurnFunction>(kNextTurn)(simulation);
    std::fprintf(stderr, "[d3probe] native NextTurn returned\n");
    return 0;
}

extern "C" __attribute__((visibility("default"))) int d3_gameplay_next_turn() {
    void* gameplay = function_at<SimulationGetter>(kGameplayGetter)();
    std::fprintf(stderr,
                 "[d3probe] starting native SIM_Gameplay::NextTurn()\n");
    function_at<NextTurnFunction>(kGameplayNextTurn)(gameplay);
    std::fprintf(stderr,
                 "[d3probe] SIM_Gameplay::NextTurn returned; worker is pending\n");
    return 0;
}

extern "C" __attribute__((visibility("default"))) int
d3_gameplay_next_turn_sync() {
    std::fprintf(stderr,
                 "[d3probe] entering native NextTurnThread synchronously\n");
    const int result = function_at<NextTurnThreadFunction>(kNextTurnThread)(
        nullptr);
    std::fprintf(stderr,
                 "[d3probe] native NextTurnThread returned (%d)\n", result);
    return result;
}

extern "C" __attribute__((visibility("default"))) float d3_read_neuron(
    const char* name) {
    void* neuron = neuron_by_name(name);
    if (neuron == nullptr) {
        std::fprintf(stderr, "[d3probe] neuron '%s' was not found\n",
                     name == nullptr ? "<null>" : name);
        return 0.0F;
    }

    const float value = *reinterpret_cast<float*>(
        reinterpret_cast<std::uintptr_t>(neuron) + kNeuronValueOffset);
    std::fprintf(stderr, "[d3probe] neuron %s = %.9g at %p\n", name, value,
                 neuron);
    return value;
}

extern "C" __attribute__((visibility("default"))) int d3_write_neuron(
    const char* name, float value) {
    void* neuron = neuron_by_name(name);
    if (neuron == nullptr) {
        std::fprintf(stderr, "[d3probe] cannot edit missing neuron '%s'\n",
                     name == nullptr ? "<null>" : name);
        return 1;
    }

    float* slot = reinterpret_cast<float*>(
        reinterpret_cast<std::uintptr_t>(neuron) + kNeuronValueOffset);
    const float before = *slot;
    *slot = value;
    std::fprintf(stderr, "[d3probe] edited neuron %s: %.9g -> %.9g\n", name,
                 before, value);
    return 0;
}

int begin_probe() {
    if (!save_game("D3_SAVE_LOADED", "d3_probe_loaded")) {
        return 1;
    }

    const std::string edited_spec(setting("D3_SAVE_EDITED", "d3_probe_edited"));
    const std::size_t first_separator = edited_spec.find("::");
    const std::size_t second_separator = edited_spec.find(
        "::", first_separator == std::string::npos ? 0 : first_separator + 2);
    if (first_separator != std::string::npos &&
        second_separator != std::string::npos) {
        const std::string edit_name =
            edited_spec.substr(first_separator + 2,
                               second_separator - first_separator - 2);
        const std::string edit_value_text = edited_spec.substr(second_separator + 2);
        char* end = nullptr;
        const float edit_value = std::strtof(edit_value_text.c_str(), &end);
        if (edit_name.empty() || end == edit_value_text.c_str() || *end != '\0') {
            std::fprintf(stderr, "[d3probe] invalid encoded edit in D3_SAVE_EDITED\n");
            return 2;
        }
        if (d3_write_neuron(edit_name.c_str(), edit_value) != 0) {
            return 3;
        }
        if (!save_game_name(edited_spec.substr(0, first_separator))) {
            return 4;
        }
    }

    const std::string turn_mode(setting("D3_SKIP_TURN", ""));
    if (turn_mode == "1" || turn_mode == "true") {
        g_probe_finished = true;
        return 0;
    }

    if (turn_mode == "sync") {
        if (d3_gameplay_next_turn_sync() != 0 ||
            !save_game("D3_SAVE_AFTER_TURN", "d3_probe_after_turn")) {
            return 6;
        }
        g_probe_finished = true;
        return 0;
    }

    if (turn_mode == "gameplay") {
        if (d3_gameplay_next_turn() != 0) {
            return 5;
        }
        g_turn_started = true;
        return 0;
    }

    if (d3_next_turn() != 0 ||
        !save_game("D3_SAVE_AFTER_TURN", "d3_probe_after_turn")) {
        return 6;
    }
    g_probe_finished = true;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int d3_run_probe() {
    g_probe_started = true;
    const int result = begin_probe();
    if (result != 0 || g_turn_started) {
        return result;
    }
    return 0;
}

int finish_gameplay_turn() {
    if (!save_game("D3_SAVE_AFTER_TURN", "d3_probe_after_turn")) {
        return 6;
    }
    g_probe_finished = true;
    return 0;
}

extern "C" __attribute__((visibility("default"))) int d3_maybe_run() {
    if (!g_load_started || g_probe_finished) {
        return 0;
    }

    const auto complete = *reinterpret_cast<volatile unsigned char*>(
        kLoadingCompleteFlag);
    if (!g_probe_started) {
        if (complete == 0) {
            return 0;
        }

        // The mainLoop breakpoint is the synchronization point.  The loader
        // worker has set the loading-screen completion flag, and this call is
        // now made on the GUI/main thread rather than racing the worker.
        g_probe_started = true;
        const int result = begin_probe();
        return result == 0 && !g_probe_finished ? 0 :
               (result == 0 ? 1 : -1);
    }

    if (g_turn_started && complete != 0) {
        const int result = finish_gameplay_turn();
        return result == 0 ? 1 : -1;
    }
    return 0;
}
