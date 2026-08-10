// Version-pinned in-process probe for Democracy 3 v1.30.2.
//
// The game is an ET_EXEC binary, so these addresses are intentionally absolute
// and must be revalidated with preflight.py whenever the binary changes.  The
// library is loaded with LD_PRELOAD, but it does not call into the game from
// its ELF constructor.  gdb invokes the exported functions after mainLoop,
// when the executable's own global initialization is complete.

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

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

constexpr std::uintptr_t kPoliticalCapitalGuard = 0xA3A0E8;
constexpr std::uintptr_t kPoliticalCapitalObject = 0xA3A0F8;
constexpr std::uintptr_t kPoliticalCapitalConstructor = 0x5FACA0;
constexpr std::uintptr_t kSpendPoliticalCapital = 0x5FADC0;

constexpr std::uintptr_t kPolicyManagerGuard = 0xA387F8;
constexpr std::uintptr_t kPolicyManagerObject = 0xA38920;
constexpr std::uintptr_t kPolicyManagerConstructor = 0x5F9F30;
constexpr std::uintptr_t kGetPolicy = 0x5FA090;
constexpr std::uintptr_t kPolicySetSlider = 0x5F5B70;
constexpr std::uintptr_t kPolicyImplement = 0x5F68F0;
constexpr std::uintptr_t kPolicyCancel = 0x5F66F0;

constexpr std::uintptr_t kPartyManagerGuard = 0xA3A4F8;
constexpr std::uintptr_t kPartyManagerObject = 0xA3A500;
constexpr std::uintptr_t kPartyManagerConstructor = 0x5F4880;
constexpr std::uintptr_t kPartyCalculateActivists = 0x5F49A0;

constexpr std::uintptr_t kPollsManagerGuard = 0xA3B288;
constexpr std::uintptr_t kPollsManagerObject = 0xA3B2A0;
constexpr std::uintptr_t kPollsManagerConstructor = 0x5FB430;
constexpr std::uintptr_t kPollsCalculateVoteRate = 0x5FB450;

constexpr std::uintptr_t kVoterManagerGuard = 0xA387D8;
constexpr std::uintptr_t kVoterManagerObject = 0xA3A5A0;
constexpr std::uintptr_t kVoterManagerConstructor = 0x620AC0;
constexpr std::uintptr_t kPreJoinParties = 0x6208E0;
constexpr std::uintptr_t kPreCalculateIncome = 0x620920;

using Constructor = void (*)(void*);
using LoadGameFunction = void (*)(void*, const std::string&);
using SaveGameFunction = void (*)(void*, const std::string&);
using SimulationGetter = void* (*)();
using NextTurnFunction = void (*)(void*);
using NextTurnThreadFunction = int (*)(void*);
using GetNeuronFunction = void* (*)(void*, std::string);
using ManagerFunction = void (*)(void*);
using GetPolicyFunction = void* (*)(void*, std::string);
using PolicyActionFunction = void (*)(void*);
using PolicySliderFunction = void (*)(void*, float);
using SpendPointsFunction = void (*)(void*, int);

bool g_load_started = false;
bool g_probe_started = false;
bool g_turn_started = false;
bool g_probe_finished = false;

struct NativeOrder {
    std::string action;
    std::string policy_name;
    float target = 0.0F;
};

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

bool parse_orders(const char* encoded, std::vector<NativeOrder>* orders) {
    if (encoded == nullptr || *encoded == '\0') {
        return true;
    }

    const std::string source(encoded);
    std::size_t start = 0;
    while (start < source.size()) {
        const std::size_t end = source.find(';', start);
        const std::string item = source.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        const std::size_t first = item.find('|');
        const std::size_t second = item.find(
            '|', first == std::string::npos ? 0 : first + 1);
        if (first == std::string::npos || second == std::string::npos ||
            item.find('|', second + 1) != std::string::npos) {
            std::fprintf(stderr, "[d3probe] malformed order '%s'\n", item.c_str());
            return false;
        }

        NativeOrder order;
        order.action = item.substr(0, first);
        order.policy_name = item.substr(first + 1, second - first - 1);
        const std::string target_text = item.substr(second + 1);
        if ((order.action != "slider" && order.action != "implement" &&
             order.action != "cancel") ||
            order.policy_name.empty() || order.policy_name.find(';') !=
                                            std::string::npos) {
            std::fprintf(stderr, "[d3probe] invalid order '%s'\n", item.c_str());
            return false;
        }

        char* parse_end = nullptr;
        order.target = std::strtof(target_text.c_str(), &parse_end);
        if (parse_end == target_text.c_str() || *parse_end != '\0' ||
            !std::isfinite(order.target) || order.target < 0.0F ||
            order.target > 1.0F) {
            std::fprintf(stderr, "[d3probe] invalid order target '%s'\n",
                         target_text.c_str());
            return false;
        }
        orders->push_back(order);
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return true;
}

int policy_integer_field(void* policy, std::size_t offset) {
    return *reinterpret_cast<int*>(reinterpret_cast<std::uintptr_t>(policy) +
                                   offset);
}

float policy_current_value(void* policy) {
    constexpr std::size_t kPolicyNeuronOffset = 0x390;
    void* neuron = *reinterpret_cast<void**>(
        reinterpret_cast<std::uintptr_t>(policy) + kPolicyNeuronOffset);
    if (neuron == nullptr) {
        return 0.0F;
    }
    return *reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(neuron) +
                                     kNeuronValueOffset);
}

bool spend_political_capital(int points) {
    ensure_object(kPoliticalCapitalGuard, kPoliticalCapitalObject,
                  kPoliticalCapitalConstructor, "SIM_PoliticalCapital");
    function_at<SpendPointsFunction>(kSpendPoliticalCapital)(
        reinterpret_cast<void*>(kPoliticalCapitalObject), points);
    return true;
}

int apply_orders(const char* encoded) {
    std::vector<NativeOrder> orders;
    if (!parse_orders(encoded, &orders) || orders.empty()) {
        return encoded == nullptr || *encoded == '\0' ? 0 : 10;
    }

    ensure_object(kPolicyManagerGuard, kPolicyManagerObject,
                  kPolicyManagerConstructor, "SIM_PolicyManager");
    void* manager = reinterpret_cast<void*>(kPolicyManagerObject);
    for (const NativeOrder& order : orders) {
        const std::string policy_name(order.policy_name);
        void* policy = function_at<GetPolicyFunction>(kGetPolicy)(
            manager, policy_name);
        if (policy == nullptr) {
            std::fprintf(stderr, "[d3probe] policy '%s' was not found\n",
                         order.policy_name.c_str());
            return 11;
        }

        if (order.action == "implement") {
            function_at<PolicyActionFunction>(kPolicyImplement)(policy);
            const int cost = policy_integer_field(policy, 0x368);
            if (!spend_political_capital(cost)) {
                return 12;
            }
            function_at<PolicySliderFunction>(kPolicySetSlider)(
                policy, order.target);
        } else if (order.action == "cancel") {
            // SIM_Policy::Cancel contains the native cancellation charge and
            // clears the manager-owned policy links itself.
            function_at<PolicyActionFunction>(kPolicyCancel)(policy);
        } else {
            const float before = policy_current_value(policy);
            const int cost = policy_integer_field(
                policy, order.target > before ? 0x370 : 0x374);
            function_at<PolicySliderFunction>(kPolicySetSlider)(
                policy, order.target);
            if (!spend_political_capital(cost)) {
                return 13;
            }
        }
        std::fprintf(stderr, "[d3probe] applied %s %s -> %.9g\n",
                     order.action.c_str(), order.policy_name.c_str(),
                     order.target);
    }
    return 0;
}

std::size_t vector_count(void* object, std::size_t begin_offset,
                         std::size_t end_offset) {
    const auto base = reinterpret_cast<std::uintptr_t>(object);
    const auto begin = *reinterpret_cast<std::uintptr_t*>(base + begin_offset);
    const auto end = *reinterpret_cast<std::uintptr_t*>(base + end_offset);
    if (begin == 0 || end < begin || (end - begin) % sizeof(void*) != 0) {
        return 0;
    }
    return (end - begin) / sizeof(void*);
}

std::size_t linked_list_count(void* object, std::size_t sentinel_offset) {
    const auto sentinel = reinterpret_cast<std::uintptr_t>(object) +
                          sentinel_offset;
    auto node = *reinterpret_cast<std::uintptr_t*>(sentinel);
    std::size_t count = 0;
    while (node != 0 && node != sentinel && count < 10000) {
        node = *reinterpret_cast<std::uintptr_t*>(node);
        ++count;
    }
    return count;
}

std::size_t voter_income_host_links(void* voter) {
    // UpdateIncome walks the inline std::list at voter + 0x128.  Its nodes
    // point at the live VoterType hosts used to accumulate income inputs.
    return linked_list_count(voter, 0x128);
}

std::ptrdiff_t party_index(void* party) {
    const auto begin = *reinterpret_cast<std::uintptr_t*>(
        kPartyManagerObject);
    const auto end = *reinterpret_cast<std::uintptr_t*>(
        kPartyManagerObject + 0x8);
    if (begin == 0 || end < begin || (end - begin) % sizeof(void*) != 0) {
        return -1;
    }
    const auto count = (end - begin) / sizeof(void*);
    for (std::size_t index = 0; index < count; ++index) {
        if (*reinterpret_cast<void**>(begin + index * sizeof(void*)) == party) {
            return static_cast<std::ptrdiff_t>(index);
        }
    }
    return -1;
}

bool refresh_manager_links() {
    ensure_object(kVoterManagerGuard, kVoterManagerObject,
                  kVoterManagerConstructor, "SIM_VoterManager");
    ensure_object(kPartyManagerGuard, kPartyManagerObject,
                  kPartyManagerConstructor, "SIM_PartyManager");
    ensure_object(kPollsManagerGuard, kPollsManagerObject,
                  kPollsManagerConstructor, "SIM_PollsManager");

    void* voters = reinterpret_cast<void*>(kVoterManagerObject);
    function_at<ManagerFunction>(kPreCalculateIncome)(voters);
    function_at<ManagerFunction>(kPreJoinParties)(voters);
    function_at<ManagerFunction>(kPartyCalculateActivists)(
        reinterpret_cast<void*>(kPartyManagerObject));
    function_at<ManagerFunction>(kPollsCalculateVoteRate)(
        reinterpret_cast<void*>(kPollsManagerObject));
    return true;
}

void write_manager_audit() {
    const char* path = std::getenv("D3_MANAGER_AUDIT_PATH");
    if (path == nullptr || *path == '\0') {
        return;
    }
    const auto voter_manager = reinterpret_cast<std::uintptr_t>(
        kVoterManagerObject);
    const std::size_t voter_types = vector_count(
        reinterpret_cast<void*>(kVoterManagerObject), 0x8, 0x10);
    const std::size_t voters = vector_count(
        reinterpret_cast<void*>(kVoterManagerObject), 0x20, 0x28);
    const std::size_t parties = vector_count(
        reinterpret_cast<void*>(kPartyManagerObject), 0x0, 0x8);
    const float poll_rate = *reinterpret_cast<float*>(
        reinterpret_cast<std::uintptr_t>(kPollsManagerObject) + 0x50);

    FILE* output = std::fopen(path, "w");
    if (output == nullptr) {
        std::fprintf(stderr, "[d3probe] cannot write manager audit '%s'\n", path);
        return;
    }
    std::fprintf(output, "format=democracy3-v1.30.2-manager-audit\n");
    std::fprintf(output, "voter_types=%zu\n", voter_types);
    std::fprintf(output, "voters=%zu\n", voters);
    std::fprintf(output, "parties=%zu\n", parties);
    std::fprintf(output, "poll_rate=%.9g\n", poll_rate);
    std::fprintf(output,
                 "refresh=PreCalculateIncome,PreJoinParties,"
                 "CalculateActivists,CalculateVoteRate\n");

    const auto party_begin = *reinterpret_cast<std::uintptr_t*>(
        kPartyManagerObject);
    for (std::size_t index = 0; index < parties; ++index) {
        void* party = *reinterpret_cast<void**>(
            party_begin + index * sizeof(void*));
        if (party == nullptr) {
            std::fprintf(output, "party[%zu].null=1\n", index);
            continue;
        }
        const auto members = linked_list_count(party, 0x10);
        const auto activists = *reinterpret_cast<int*>(
            reinterpret_cast<std::uintptr_t>(party) + 0x38);
        const auto activist_fraction = *reinterpret_cast<float*>(
            reinterpret_cast<std::uintptr_t>(party) + 0x30);
        const auto status = *reinterpret_cast<int*>(
            reinterpret_cast<std::uintptr_t>(party) + 0x28);
        std::fprintf(output,
                     "party[%zu].members=%zu activists=%d "
                     "activist_fraction=%.9g status=%d\n",
                     index, members, activists, activist_fraction, status);
    }

    const auto begin = *reinterpret_cast<std::uintptr_t*>(voter_manager + 0x20);
    for (std::size_t index = 0; index < voters; ++index) {
        void* voter = *reinterpret_cast<void**>(begin + index * sizeof(void*));
        const auto voter_address = reinterpret_cast<std::uintptr_t>(voter);
        const auto linked_party = voter == nullptr
                                      ? static_cast<void*>(nullptr)
                                      : *reinterpret_cast<void**>(
                                            voter_address + 0x168);
        std::fprintf(output,
                     "voter[%zu].party_index=%td income_host_links=%zu\n",
                     index, party_index(linked_party),
                     voter == nullptr ? 0 : voter_income_host_links(voter));
    }
    std::fclose(output);
    std::fprintf(stderr, "[d3probe] wrote manager audit '%s'\n", path);
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

bool setting_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr &&
           (std::string(value) == "1" || std::string(value) == "true");
}

int capture_turn_count() {
    const char* value = std::getenv("D3_TURN_COUNT");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long count = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || count < 1 || count > 100) {
        return -1;
    }
    return static_cast<int>(count);
}

std::string capture_save_name(const std::string& prefix, int turn) {
    return prefix + "_turn" + std::to_string(turn);
}

int begin_probe() {
    if (setting_enabled("D3_MANAGER_AUDIT")) {
        if (!refresh_manager_links()) {
            return 20;
        }
    }
    if (!save_game("D3_SAVE_LOADED", "d3_probe_loaded")) {
        return 1;
    }
    if (setting_enabled("D3_MANAGER_AUDIT")) {
        write_manager_audit();
        const char* manager_save = std::getenv("D3_SAVE_MANAGERS");
        if (manager_save != nullptr && *manager_save != '\0' &&
            !save_game_name(manager_save)) {
            return 21;
        }
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

    const char* single_orders = std::getenv("D3_ORDERS");
    const int turn_count = capture_turn_count();
    if (turn_count < 0) {
        std::fprintf(stderr, "[d3probe] D3_TURN_COUNT must be in [1, 100]\n");
        return 22;
    }
    const std::string turn_mode(setting("D3_SKIP_TURN", ""));
    if (turn_mode == "1" || turn_mode == "true") {
        if (apply_orders(single_orders) != 0) {
            return 10;
        }
        const char* orders_save = std::getenv("D3_SAVE_ORDERS");
        if (orders_save != nullptr && *orders_save != '\0' &&
            !save_game_name(orders_save)) {
            return 23;
        }
        g_probe_finished = true;
        return 0;
    }

    if (turn_mode == "sync") {
        if (turn_count > 0) {
            const std::string prefix(
                setting("D3_CAPTURE_PREFIX", "d3_probe_capture"));
            for (int index = 0; index < turn_count; ++index) {
                const std::string variable =
                    "D3_ORDERS_" + std::to_string(index);
                if (apply_orders(std::getenv(variable.c_str())) != 0 ||
                    d3_gameplay_next_turn_sync() != 0 ||
                    !save_game_name(capture_save_name(prefix, index + 1))) {
                    return 24;
                }
            }
            g_probe_finished = true;
            return 0;
        }
        if (apply_orders(single_orders) != 0) {
            return 10;
        }
        const char* orders_save = std::getenv("D3_SAVE_ORDERS");
        if (orders_save != nullptr && *orders_save != '\0' &&
            !save_game_name(orders_save)) {
            return 23;
        }
        if (d3_gameplay_next_turn_sync() != 0 ||
            !save_game("D3_SAVE_AFTER_TURN", "d3_probe_after_turn")) {
            return 6;
        }
        g_probe_finished = true;
        return 0;
    }

    if (turn_mode == "gameplay") {
        if (turn_count > 0) {
            std::fprintf(stderr,
                         "[d3probe] bounded captures require sync turn mode\n");
            return 25;
        }
        if (apply_orders(single_orders) != 0) {
            return 10;
        }
        if (d3_gameplay_next_turn() != 0) {
            return 5;
        }
        g_turn_started = true;
        return 0;
    }

    if (turn_count > 0) {
        std::fprintf(stderr,
                     "[d3probe] bounded captures require --sync mode\n");
        return 25;
    }
    if (apply_orders(single_orders) != 0 || d3_next_turn() != 0 ||
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
