#include "mlsys.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace absl;
using json = nlohmann::json;

namespace mlsys {

StatusOr<Problem> ReadProblem(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return absl::NotFoundError("File not found: " + filename);
    }

    json data;
    try {
        data = json::parse(file);
    } catch (const json::parse_error& e) {
        return absl::InvalidArgumentError(std::string("JSON parse error: ") + e.what());
    }

    Problem problem;

    try {
        // Parse Tensors
        const auto& widths = data.at("widths");
        const auto& heights = data.at("heights");
        
        if (widths.size() != heights.size()) {
            return absl::InvalidArgumentError("widths and heights arrays must be the same size");
        }
        
        for (size_t i = 0; i < widths.size(); ++i) {
            problem.tensors.push_back(Tensor{
                widths[i].get<Width>(), 
                heights[i].get<Height>()
            });
        }

        // Parse Operations
        const auto& inputs = data.at("inputs");
        const auto& outputs = data.at("outputs");
        const auto& base_costs = data.at("base_costs");
        const auto& op_types = data.at("op_types");

        if (inputs.size() != outputs.size() || 
            inputs.size() != base_costs.size() || 
            inputs.size() != op_types.size()) {
            return absl::InvalidArgumentError("Op arrays must all be the same size");
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            Op op;
            op.op_type = op_types[i].get<OpType>();
            op.inputs = inputs[i].get<Inputs>();
            op.outputs = outputs[i].get<Outputs>();
            op.base_cost = base_costs[i].get<BaseCost>();
            problem.ops.push_back(op);
        }

        // Parse System Constraints
        problem.fast_memory_capacity = data.at("fast_memory_capacity").get<FastMemoryCapacity>();
        problem.slow_memory_bandwidth = data.at("slow_memory_bandwidth").get<SlowMemoryBandwidth>();

        // Parse Granularity 
        const auto& gran = data.at("native_granularity");
        if (gran.size() >= 2) {
            problem.native_granularity.width = gran[0].get<Width>();
            problem.native_granularity.height = gran[1].get<Height>();
            problem.native_granularity.depth = 1.get<Depth>();
        } else {
            return absl::InvalidArgumentError("native_granularity must have at least 2 elements");
        }

    } catch (const json::type_error& e) {
        return absl::InvalidArgumentError(std::string("JSON type error: ") + e.what());
    } catch (const json::out_of_range& e) {
        return absl::InvalidArgumentError(std::string("JSON missing expected field: ") + e.what());
    }

    return problem;
}

StatusOr<Solution> ReadSolution(const std::string& filename) {
    // Placeholder implementation
    return UnimplementedError("ReadSolution not yet implemented");
}

StatusOr<TotalLatency> Evaluate(const Problem& problem, const Solution& solution) {
    // Placeholder implementation
    return UnimplementedError("Evaluate not yet implemented");
}

absl::StatusOr<Solution> Solve(const Problem& problem) {
    // Placeholder implementation
    return UnimplementedError("Solve not yet implemented");
}

}  // namespace mlsys