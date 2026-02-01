#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void doTrace(T* input, T* output, size_t rows, size_t cols, size_t size) {
    extern __shared__ unsigned char smem_raw[];
    T* s_mem = reinterpret_cast<T*>(smem_raw);
    size_t tid = threadIdx.x;
    size_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_tid < size) {
        s_mem[tid] = input[global_tid];
    } else {
        s_mem[tid] = 0;
    }
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

    std::vector<T> h_diag(size);
    for (size_t i = 0; i < size; ++i) {
        h_diag[i] = h_input[i * cols + i];
    }

    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);
    size_t s_mem_size = block_dim.x * sizeof(T);

    T* d_output = nullptr;
    T* d_input = nullptr;
    T h_result = 0;

    RUNTIME_CHECK(cudaMalloc(&d_input, bytes));
    RUNTIME_CHECK(cudaMalloc(&d_output, sizeof(T)));
    RUNTIME_CHECK(cudaMemcpy(d_input, h_diag.data(), bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_output, 0, sizeof(T)));

    // 调用核函数
    doTrace<T><<<grid_dim, block_dim, s_mem_size>>>(d_input, d_output, rows, cols, size);

    RUNTIME_CHECK(cudaMemcpy(&h_result, d_output, sizeof(T), cudaMemcpyDeviceToHost));
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_output));

    return h_result;
}

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return (float)x;
}

template <>
__device__ __forceinline__ float to_float<half>(half x) {
    return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    return (T)x;
}

template <>
__device__ __forceinline__ half from_float<half>(float x) {
    return __float2half_rn(x);
}
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    // 这里的 mask 0xffffffff 代表 Warp 中所有 32 个线程都参与
    // 这种异或洗牌方式比单纯的 down shuffle 兼容性更好，不需要必须是2的幂次逻辑
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
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
//__global__ void flash_attn_kernel(
//        const T* __restrict__ Q,
//        const T* __restrict__ K,
//        const T* __restrict__ V,
//        T* __restrict__ O,
//        int batch_size,
//        int target_seq_len,
//        int src_seq_len,
//        int query_heads,
//        int kv_heads,
//        int head_dim,
//        bool is_causal
//) {
//    int idx = blockIdx.x;
//    int tid = threadIdx.x;
//    int total = batch_size * target_seq_len * query_heads;
//    if (idx >= total) return;
//
//    extern __shared__ float smem[];
//
//    int qh = idx % query_heads;
//    int t  = (idx / query_heads) % target_seq_len;
//    int b  = idx / (query_heads * target_seq_len);
//
//    int kv_h = qh * kv_heads / query_heads;
//
//    float q = 0.0f;
//    if (tid < head_dim) {
//        const T* q_ptr = Q + (((b * target_seq_len + t) * query_heads + qh) * head_dim);
//        q = to_float(q_ptr[tid]);
//    }
//
//    float m = -INFINITY; // 老的最大值
//    float l = 0.0f; // 归一化分母
//    float o = 0.0f; // 未归一化输出
//    const float scale = rsqrtf((float)head_dim);
//
//    for (int s = 0; s < src_seq_len; ++s) {
//        if (is_causal && s > t) break; // 因果注意力
//
//        const T* k_ptr = K + (((b * src_seq_len + s) * kv_heads + kv_h) * head_dim);
//        const T* v_ptr = V + (((b * src_seq_len + s) * kv_heads + kv_h) * head_dim);
//
//        __shared__ float score; // head_dim计算处理的attention分数
//        if (tid == 0) score = 0.0f;
//        __syncthreads(); // 每个线程先取值
//
//        float k = 0.0f;
//        if (tid < head_dim) {
//            k = to_float(k_ptr[tid]);
//        }
//        smem[tid] = q * k;
//
//        __syncthreads();
//        for (int i = blockDim.x / 2; i > 0; i >>= 1) {
//            if (tid < i) {
//                smem[tid] += smem[tid + i];
//            }
//            __syncthreads();
//        }
//
//        if (tid == 0) {
//            score = smem[0];
//        }
//        __syncthreads();
//
//        float s_val = score * scale;
//
//        float m_new = fmaxf(m, s_val); // 新的最大值，每个tid上都是一样的
//        float alpha = expf(m - m_new); // 老的对新的折算比率
//        float beta  = expf(s_val - m_new);
//
//        float v = 0.0f;
//        if (tid < head_dim) {
//            v = to_float(v_ptr[tid]);
//        }
//        o = o * alpha + beta * v; // 一个个v值计算的，而不是整个向量
//        l = l * alpha + beta;
//        m = m_new;
//
//        __syncthreads(); // 当前这个online计算完成
//    }
//
//    if(tid < head_dim) {
//        O[(((b * target_seq_len + t) * query_heads + qh) * head_dim) + tid] = from_float<T>(o / l);
//    }
//}
__global__ void flash_attn_kernel(
        const T* __restrict__ Q,
        const T* __restrict__ K,
        const T* __restrict__ V,
        T* __restrict__ O,
        int batch_size,
        int target_seq_len,
        int src_seq_len,
        int query_heads,
        int kv_heads,
        int head_dim,
        bool is_causal
) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int total = batch_size * target_seq_len * query_heads;
    if (idx >= total) return;

    extern __shared__ float smem[];

    int qh = idx % query_heads;
    int t  = (idx / query_heads) % target_seq_len;
    int b  = idx / (query_heads * target_seq_len);

    int kv_h = qh * kv_heads / query_heads;

    float q = 0.0f;
    if (tid < head_dim) {
        const T* q_ptr = Q + (((b * target_seq_len + t) * query_heads + qh) * head_dim);
        q = to_float(q_ptr[tid]);
    }

    float m = -INFINITY; // 老的最大值
    float l = 0.0f; // 归一化分母
    float o = 0.0f; // 未归一化输出
    const float scale = rsqrtf((float)head_dim);

    for (int s = 0; s < src_seq_len; ++s) {
        if (is_causal && s > t) break; // 因果注意力

        const T* k_ptr = K + (((b * src_seq_len + s) * kv_heads + kv_h) * head_dim);
        const T* v_ptr = V + (((b * src_seq_len + s) * kv_heads + kv_h) * head_dim);

        float k = 0.0f;
        if (tid < head_dim) {
            k = to_float(k_ptr[tid]);
        }
        float val = q * k;
        // Warp 内归约: 得到每个 warp 的和
        val = warpReduceSum(val);

        int lane = tid % 32;
        int warp_id = tid / 32;

        if (lane == 0) smem[warp_id] = val;
        __syncthreads(); // 等待所有 warp 写完

        // 让第一个 warp (warp 0) 将所有 warp 的结果加起来
        float score = 0.0f;
        if (warp_id == 0) {
            // 假设 blockDim 不超过 1024 (即 warp 数量 <= 32)
            // 只有 warp 0 的前 (blockDim/32) 个线程需要读取 smem
            val = (tid < (blockDim.x / 32)) ? smem[lane] : 0.0f;
            val = warpReduceSum(val); // 再次 warp 归约得到最终总和
            if (lane == 0) score = val;
        }
        __syncthreads();
        if (tid == 0) smem[0] = score; // 复用 smem[0] 广播
        __syncthreads();
        score = smem[0];

        float s_val = score * scale;
        float m_prev = m;
        m = fmaxf(m_prev, s_val);

        float s_part = expf(s_val - m);
        float m_part = expf(m_prev - m); // 如果 m 更新了，这里是 < 1 的数

        // 更新分母
        l = l * m_part + s_part;

        // 5. 加载 Value 并更新 Output
        float v = 0.0f;
        if (tid < head_dim) {
            v = to_float(v_ptr[tid]);
        }

        // 更新分子 (Vector update)
        o = o * m_part + s_part * v;
    }

    if(tid < head_dim) {
        O[(((b * target_seq_len + t) * query_heads + qh) * head_dim) + tid] = from_float<T>(o / l);
    }
}
// 扩展到大于等于n的最小2的幂（n > 0）
unsigned int nextPowerOfTwo(unsigned int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    // TODO: Implement the flash attention function
    const T scale = static_cast<T>(1.0 / std::sqrt(head_dim));
    size_t q_size = h_q.size() * sizeof(T);
    size_t k_size = h_k.size() * sizeof(T);
    size_t v_size = h_v.size() * sizeof(T);
    size_t o_size = h_o.size() * sizeof(T);

    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_size));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_size));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size));

    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice));

    int blocks = batch_size * target_seq_len * query_heads;
    int threads = head_dim;
//    threads = nextPowerOfTwo(head_dim);
    size_t smen_size = threads * sizeof(float );
    smen_size = smen_size / 32;

    flash_attn_kernel<<<blocks, threads, smen_size>>>(
            d_q, d_k, d_v, d_o,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            is_causal
    );

    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost));

    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));

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
