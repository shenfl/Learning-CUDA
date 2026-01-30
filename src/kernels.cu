#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ doTrace(T* input, T* output, size_t rows, size_t cols, size_t size) {
    extern __shared__ T s_mem[];
    size_t tid = threadIdx.x;
    size_t idx;
    s_mem[tid] = input[cols * tid + tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mem[tid] += s_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, s_mem[0]);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // TODO: Implement the trace function
    size_t size = 0;
    if(rows < cols) {
        size = rows;
    } else {
        size = cols;
    }
    size_t bytes = sizeof(T) * size;
    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);
    size_t s_mem_size = block_dim.x * sizeof(T);

    T* d_output = nullptr;
    T* d_input = nullptr;
    T h_result = 0;

    RUNTIME_CHECK(cudaMalloc(&d_input, bytes));
    RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_output, 0, sizeof(T)));

    // 调用核函数
    doTrace<T><<<grid_dim, block_dim, s_mem_size>>>(d_input, d_output, rows, cols, size);

    RUNTIME_CHECK(cudaMemcpy(&h_result, d_output, sizeof(T), cudaMemcpyDeviceToHost));
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return h_result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
