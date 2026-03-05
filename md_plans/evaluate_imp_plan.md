# Implementation Plan for Evaluate Function

## Proposed Changes

### [mlsys.cpp](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.cpp)
I will implement `mlsys::Evaluate(const Problem& problem, const Solution& solution)`:
1.  **Initialize**: `total_latency = 0`. Keep track of `resident_tensors` (tensors currently in fast memory, initialized to empty, but populated by graph inputs when first needed, or just track what was left by previous subgraphs).
2.  **Iterate Subgraphs**: For each subgraph in the solution:
    *   **Determine shapes per step**: For a given tile configuration [(w, h, k)](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.h#55-62):
        *   Determine the output spatial shape `sub_w x sub_h` corresponding to the current traversal tile.
        *   Propagate shapes backward from outputs to inputs. Subgraph outputs have shape `sub_w x sub_h`.
        *   For `Pointwise`: input required is the exact same bounding box as the requested output.
        *   For `MatMul(LHS, RHS)`: if it's the target of `k` split, LHS required is `height = sub_h`, `width = current k_start to k_end`. RHS required is `height = current k_start to k_end`, `width = sub_w`.
        *   If `MatMul(LHS, RHS)` is NOT the target of `k` split (it fully reduces to provide an intermediate), then `LHS required width = FULL K`, `RHS required height = FULL K`.
        *   Compute the computation time for this step using the exact mathematical area:
            *   `padding_ratio = (max(W_native, subgraph_w) * max(H_native, subgraph_h)) / (subgraph_w * subgraph_h)`
            *   *(Note: The `padding_ratio` ensures that if we choose a granularity smaller than the native hardware shape e.g., 64x64 on 128x128 native, we still pay the full compute cost of the 128x128 hardware execution, because the compute cores are forced to pad the inputs with zeros. This matches Example 1 Strategy C.)*
            *   `step_compute  = SUM(Op_base_cost * (req_w * req_h * req_k) / (Full_W * Full_H * Full_K) * padding_ratio)`
    *   **Iterate Tiles**: Determine the number of spatial tiles `num_w = ceil(W_out/w)`, `num_h = ceil(H_out/h)`, and reduction steps `num_k = ceil(K_full/k)` (if a MatMul produces the subgraph output; otherwise 1).
    *   Initialize `currently_resident_slices = {}` (empty) to track intra-subgraph residency. Wait, actually, any full tensors specified in `tensors_to_retain` from *previous* subgraphs initially reside here.
    *   Use `traversal_order` to iterate over spatial tiles [(tile_x, tile_y)](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.h#55-62).
    *   For each spatial tile:
        *   **Output Stationary Residency**: The ultimate output slice(s) for this spatial tile of the subgraph are IMMEDIATELY locked into `currently_resident_slices` as accumulators. They cannot be evicted until the entire spatial tile (all `k` steps and all operations) finishes.
        *   Iterate over `k` steps.
        *   **Memory In/Out**:
            *   Calculate the exact `required_slices` (bounding boxes) for this step across all operations.
            *   Ensure the output accumulator slice(s) is also considered required.
            *   `memory_in = sum of sizes of (required_slices - currently_resident_slices)`
            *   Update `currently_resident_slices = required_slices` (slices not needed in this step are implicitly evicted by hardware, except the locked output accumulators).
            *   Any subgraph output slice must be written back at the end of its reduction steps. `memory_out = size of output slice` (if it's the last k-step, or if Pointwise).
        *   **Latency per step**: `max(step_compute, (memory_in + memory_out) / Bandwidth)`. Sum up latencies for the subgraph. Check if `required_slices` total memory exceeds `fast_memory_capacity` (`OOM`).
    *   Sum up latencies for the subgraph. Add to `total_latency`. Ensure it matches `subgraph_latency`.
    *   **Residency Update**: At the end of the subgraph, retain only tensors specified in `tensors_to_retain`. Evict the rest.

## Verification Plan

### Automated Tests
- Build using CMake (`mkdir build && cd build && cmake .. && make`).
- Run the executable on all test cases in the `examples/` directory.
- Verify that the tool prints `Done.` without errors, and the computed subgraph latencies match exactly with those provided in the solution (`example-X-output-Y.json`).
- If missing a tester in [main.cpp](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/main.cpp), I will write a simple test harness in [main.cpp](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/main.cpp) or a [evaluate_test.cpp](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/evaluate_test.cpp) to load [Problem](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.h#70-78) and [Solution](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.h#90-94), call [Evaluate](file:///home/linux_wsl/UBC_Local/UBC_Year_4/CPEN_511/MLSys2026-Google-Graph-Scheduling-Competition/mlsys.cpp#160-164), and check the returned `TotalLatency`.
