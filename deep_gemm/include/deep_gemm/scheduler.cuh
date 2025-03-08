#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks; // m维度上的分块数量

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;

    // For grouped GEMM
    int* grouped_layout;
    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m,
                                                  int* grouped_layout = nullptr) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
    }

    // 使用 grouped ordering, 提高L2 cache命中率
    // 参考 triton tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        // num_m_blocks: m方向的block数量
        // kNumNBlocksPerGroup: 每个group中的N block数量
        // num_blocks_per_group: 每个group中的block数量 (这里只在N方向划分group, m方向上不划分group)
        auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;
        // group_idx: 当前group的索引
        auto group_idx = block_idx / num_blocks_per_group;
        // first_n_block_idx: 当前group中第一个block在N方向上的索引
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
        // num_n_blocks_in_group: 当前group中N block的数量
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
        // in_group_idx: 当前block在group中的索引
        auto in_group_idx = block_idx % num_blocks_per_group;
        // m_block_idx: 当前block在group内的m方向索引, 由于不在m方向划分group, 因此也是全局的m索引
        m_block_idx = in_group_idx / num_n_blocks_in_group;
        // n_block_idx: 当前block的全局n索引
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(grouped_layout + m_block_idx * BLOCK_M);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // 采用persistent kernel, gridDim.x固定是num_sms
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};
#pragma clang diagnostic pop

} // namespace deep_gemm
