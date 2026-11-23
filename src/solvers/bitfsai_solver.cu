// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/bitfsai_solver.h>
#include <basic_types.h>
#include <cutil.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>

namespace amgx
{

namespace bitfsai_detail
{

constexpr int MAX_PATCH = 16;

// Per-device exclusive staging workspace for 4-bit nibble packing.
// The returned Lease holds the device mutex through kernel completion so
// concurrent solvers on the same or different GPUs cannot alias or free buffers.
struct DeviceStagingSlot
{
    std::mutex mtx;
    uint8_t* ptr = nullptr;
    size_t   cap = 0;

    ~DeviceStagingSlot()
    {
        if (ptr)
        {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
    }
};

class PerDeviceStagingPool
{
    std::mutex m_map_mtx;
    std::map<int, std::unique_ptr<DeviceStagingSlot>> m_slots;

    DeviceStagingSlot& slot_for(int dev)
    {
        std::lock_guard<std::mutex> lk(m_map_mtx);
        auto &entry = m_slots[dev];
        if (!entry) entry = std::make_unique<DeviceStagingSlot>();
        return *entry;
    }

public:
    class Lease
    {
        DeviceStagingSlot* m_slot = nullptr;
        std::unique_lock<std::mutex> m_lk;
        uint8_t* m_ptr = nullptr;
        bool m_release_on_exit = false;

    public:
        Lease() = default;
        Lease(DeviceStagingSlot &slot, size_t bytes, bool release_on_exit)
            : m_slot(&slot), m_lk(slot.mtx), m_release_on_exit(release_on_exit)
        {
            if (bytes > slot.cap)
            {
                if (slot.ptr) cudaFree(slot.ptr);
                slot.ptr = nullptr;
                slot.cap = 0;
                if (cudaMalloc(&slot.ptr, bytes) == cudaSuccess)
                {
                    slot.cap = bytes;
                }
            }
            m_ptr = slot.ptr;
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        uint8_t* data() const { return m_ptr; }

        void finish_and_unlock(cudaStream_t st)
        {
            if (!m_slot) return;
            cudaStreamSynchronize(st);
            if (m_release_on_exit && m_slot->ptr)
            {
                cudaFree(m_slot->ptr);
                m_slot->ptr = nullptr;
                m_slot->cap = 0;
            }
            m_ptr = nullptr;
            m_slot = nullptr;
            if (m_lk.owns_lock()) m_lk.unlock();
        }

        ~Lease() { finish_and_unlock(0); }
    };

    Lease acquire(size_t bytes, bool release_on_exit)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        return Lease(slot_for(dev), bytes, release_on_exit);
    }

    void release_all()
    {
        std::lock_guard<std::mutex> map_lk(m_map_mtx);
        for (auto &kv : m_slots)
        {
            if (!kv.second) continue;
            std::lock_guard<std::mutex> slot_lk(kv.second->mtx);
            if (kv.second->ptr)
            {
                cudaFree(kv.second->ptr);
                kv.second->ptr = nullptr;
                kv.second->cap = 0;
            }
        }
        m_slots.clear();
    }
};

static PerDeviceStagingPool g_staging_pool;

// Exact FP4 (E2M1) magnitude lattice normalized by 6.0 (bit 3 = sign)
__device__ __constant__ float c_nvfp4_lut[16] = {
    0.0f / 6.0f,  0.5f / 6.0f,  1.0f / 6.0f,  1.5f / 6.0f,
    2.0f / 6.0f,  3.0f / 6.0f,  4.0f / 6.0f,  6.0f / 6.0f,
   -0.0f / 6.0f, -0.5f / 6.0f, -1.0f / 6.0f, -1.5f / 6.0f,
   -2.0f / 6.0f, -3.0f / 6.0f, -4.0f / 6.0f, -6.0f / 6.0f
};

__device__ __forceinline__ uint8_t encode_nvfp4(double r)
{
    uint8_t sgn = (r < 0.0) ? 8u : 0u;
    double ar = fabs(r) * 6.0;
    uint8_t mag = 0u;
    if      (ar < 0.25) mag = 0u;
    else if (ar < 0.75) mag = 1u;
    else if (ar < 1.25) mag = 2u;
    else if (ar < 1.75) mag = 3u;
    else if (ar < 2.50) mag = 4u;
    else if (ar < 3.50) mag = 5u;
    else if (ar < 5.00) mag = 6u;
    else                mag = 7u;
    return (mag == 0u) ? 0u : (sgn | mag);
}

__device__ __forceinline__ uint8_t encode_int4(double r)
{
    int q = (int)llround(r * 7.0);
    if (q > 7) q = 7;
    if (q < -7) q = -7;
    return (uint8_t)(q + 8); // [1..15], 8 encodes exact 0
}

__device__ __forceinline__ float decode_nibble(uint8_t nib, int mode)
{
    if (mode == 0) return c_nvfp4_lut[nib & 0x0F];
    return (float)((int)(nib & 0x0F) - 8) * (1.0f / 7.0f);
}

template <typename T>
__device__ __forceinline__ double csr_lookup_fast(const int* __restrict__ rp,
                                                  const int* __restrict__ ci,
                                                  const T* __restrict__ val,
                                                  int row, int col)
{
    int lo = rp[row], hi = rp[row + 1];
    for (int k = lo; k < hi; ++k) if (ci[k] == col) return (double)val[k];
    return 0.0;
}

__device__ __forceinline__ int csr_find_pos_fast(const int* __restrict__ rp,
                                                 const int* __restrict__ ci,
                                                 int row, int col)
{
    int lo = rp[row], hi = rp[row + 1];
    for (int k = lo; k < hi; ++k) if (ci[k] == col) return k;
    return -1;
}

// Probe local patch A-energy-norm error of split INT4 vs split E2M1 for AUTO selector
template <typename T>
__global__ void bitfsai_energy_probe_kernel(
    int n, int max_patch,
    const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    double* __restrict__ err_totals)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int stride = (n > 8192) ? (n / 8192) : 1;
    if ((i % stride) != 0) return;

    int rs = rp[i], re = rp[i + 1];
    int    J[MAX_PATCH];
    double Wabs[MAX_PATCH];
    int m_low = 0, max_low = max_patch - 1;
    double a_ii = 1.0;

    for (int k = rs; k < re; ++k)
    {
        int col = ci[k];
        double v = (double)aval[k];
        if (col == i) a_ii = v;
        else if (col < i)
        {
            double av = fabs(v);
            if (m_low < max_low) { J[m_low] = col; Wabs[m_low] = av; m_low++; }
            else if (max_low > 0)
            {
                int mi = 0; double mw = Wabs[0];
                for (int t = 1; t < max_low; ++t) if (Wabs[t] < mw) { mw = Wabs[t]; mi = t; }
                if (av > mw) { J[mi] = col; Wabs[mi] = av; }
            }
        }
    }
    if (m_low == 0) return;
    int m = m_low + 1;
    J[m - 1] = i;

    double L[MAX_PATCH * MAX_PATCH];
    for (int a = 0; a < m; ++a)
    {
        for (int b = 0; b <= a; ++b)
        {
            double sum = (a == m - 1 && b == m - 1) ? a_ii : csr_lookup_fast(rp, ci, aval, J[a], J[b]);
            for (int k = 0; k < b; ++k) sum -= L[a * MAX_PATCH + k] * L[b * MAX_PATCH + k];
            if (a == b)
            {
                if (!(sum > 0.0) || !(sum == sum && sum >= -DBL_MAX && sum <= DBL_MAX)) return;
                L[a * MAX_PATCH + a] = sqrt(sum);
            }
            else L[a * MAX_PATCH + b] = sum / L[b * MAX_PATCH + b];
        }
    }
    double g[MAX_PATCH];
    double llast = L[(m - 1) * MAX_PATCH + (m - 1)];
    if (!(llast > 0.0) || !(llast == llast && llast >= -DBL_MAX && llast <= DBL_MAX)) return;
    g[m - 1] = 1.0 / llast;
    for (int a = m - 2; a >= 0; --a)
    {
        double s = 0.0;
        for (int b = a + 1; b < m; ++b) s += L[b * MAX_PATCH + a] * g[b];
        g[a] = -s / L[a * MAX_PATCH + a];
    }
    if (!(g[m - 1] > 0.0) || !(g[m - 1] == g[m - 1] && g[m - 1] >= -DBL_MAX && g[m - 1] <= DBL_MAX)) return;
    double omax = 0.0;
    for (int a = 0; a < m - 1; ++a) omax = fmax(omax, fabs(g[a]));
    if (!(omax > 0.0) || !(omax == omax && omax >= -DBL_MAX && omax <= DBL_MAX)) return;

    double d_i4[MAX_PATCH], d_e2[MAX_PATCH];
    d_i4[m - 1] = 0.0; d_e2[m - 1] = 0.0;
    for (int a = 0; a < m - 1; ++a)
    {
        uint8_t c4 = encode_int4(g[a] / omax);
        uint8_t ce = encode_nvfp4(g[a] / omax);
        d_i4[a] = g[a] - omax * (double)decode_nibble(c4, 4);
        d_e2[a] = g[a] - omax * (double)decode_nibble(ce, 0);
    }
    double e_i4 = 0.0, e_e2 = 0.0;
    for (int b = 0; b < m; ++b)
    {
        double s4 = 0.0, se = 0.0;
        for (int a = b; a < m; ++a)
        {
            double lab = L[a * MAX_PATCH + b];
            s4 += lab * d_i4[a];
            se += lab * d_e2[a];
        }
        e_i4 += s4 * s4;
        e_e2 += se * se;
    }
    atomicAdd(&err_totals[0], e_i4);
    atomicAdd(&err_totals[1], e_e2);
}

// 100% GPU-parallel Zero-Duplicate-Index FSAI(1) Numeric Setup Kernel.
constexpr int kErrMissingDiag      = 1;
constexpr int kErrNonsymmetric     = 2;
constexpr int kErrInvalidDiag      = 4;
constexpr int kErrUnrepresentable  = 8;
constexpr int kFlagWideApply       = 16;
constexpr double kEnergyTol        = 1.0e-4;

__device__ __forceinline__ bool is_finite_f64(double v)
{
    return v == v && v >= -DBL_MAX && v <= DBL_MAX;
}

__device__ __forceinline__ bool is_storable_f32(double v)
{
    float s = (float)v;
    return is_finite_f64(v) && s == s && s != 0.0f && s >= -FLT_MAX && s <= FLT_MAX;
}

enum class RowProjStatus : uint8_t
{
    Applied,
    Fallback,
    InvalidInput,
    InvalidStorage
};

// Cancellation-free 2x2 A-energy re-projection with Kahan-compensated stored-FP32 energy validation.
template <typename T>
__device__ __forceinline__ RowProjStatus project_row_device(
    int m, const double* q_dec, const double* L,
    const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    const int* J, double* diag_out, double* scale_out)
{
    if (m < 2) return RowProjStatus::Fallback;
    double c22 = csr_lookup_fast(rp, ci, aval, J[m - 1], J[m - 1]);
    if (!(c22 > 0.0) || !is_finite_f64(c22)) return RowProjStatus::InvalidInput;

    double c11 = 0.0, c12 = 0.0;
    for (int b = 0; b < m - 1; ++b)
    {
        double ub = 0.0;
        for (int a = b; a < m - 1; ++a) ub += L[a * MAX_PATCH + b] * q_dec[a];
        c11 += ub * ub;
        c12 += ub * L[(m - 1) * MAX_PATCH + b];
    }
    if (!(c11 > 0.0) || !is_finite_f64(c11) || !is_finite_f64(c12)) return RowProjStatus::Fallback;

    double projection = c12 / c11;
    if (!is_finite_f64(projection)) return RowProjStatus::Fallback;

    // Orthogonal residual avoids catastrophic cancellation in c11 * c22 - c12^2
    double residual_sq = 0.0;
    for (int b = 0; b < m; ++b)
    {
        double ub = 0.0;
        if (b < m - 1)
        {
            for (int a = b; a < m - 1; ++a) ub += L[a * MAX_PATCH + b] * q_dec[a];
        }
        double res = L[(m - 1) * MAX_PATCH + b] - projection * ub;
        residual_sq += res * res;
    }
    if (!(residual_sq > 0.0) || !is_finite_f64(residual_sq)) return RowProjStatus::Fallback;

    double g_opt = 1.0 / sqrt(residual_sq);
    double s_opt = -projection * g_opt;
    if (!(g_opt > 0.0) || !is_finite_f64(g_opt) || !is_finite_f64(s_opt)) return RowProjStatus::Fallback;

    float g_f32 = (float)g_opt;
    float s_f32 = (float)s_opt;
    if (s_f32 == 0.0f) return RowProjStatus::Fallback;
    if (!is_storable_f32(g_opt) || !is_storable_f32(s_opt)) return RowProjStatus::InvalidStorage;

    double sg = (double)g_f32;
    double ss = (double)s_f32;
    double energy = 0.0, comp = 0.0, sum_abs = 0.0;
    for (int a = 0; a < m; ++a)
    {
        double wa = (a == m - 1) ? sg : ss * q_dec[a];
        for (int b = 0; b < m; ++b)
        {
            double wb = (b == m - 1) ? sg : ss * q_dec[b];
            double term = wa * csr_lookup_fast(rp, ci, aval, J[a], J[b]) * wb;
            if (!is_finite_f64(term)) return RowProjStatus::InvalidStorage;
            double adj = term - comp;
            double next = energy + adj;
            comp = (next - energy) - adj;
            energy = next;
            sum_abs += fabs(term);
        }
    }
    if (!is_finite_f64(energy) || !is_finite_f64(sum_abs)) return RowProjStatus::InvalidStorage;
    double bound = 8.0 * DBL_EPSILON * sum_abs;
    if (!(fabs(energy - 1.0) + bound <= kEnergyTol)) return RowProjStatus::Fallback;

    *diag_out = g_opt;
    *scale_out = s_opt;
    return RowProjStatus::Applied;
}

// Local principal-submatrix Cholesky always runs in FP64 regardless of V-cycle precision.
template <typename T>
__global__ void bitfsai_build_kernel(
    int n, int mode, int max_patch, bool use_reproject,
    const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    float* __restrict__ g_diag,
    float* __restrict__ g_off_scale,
    double* __restrict__ g_diag_f64,
    uint8_t* __restrict__ q_tmp,
    __half* __restrict__ g_f16,
    double* __restrict__ g_f64,
    int* __restrict__ err_flags,
    int* __restrict__ diag_fallback_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int rs = rp[i], re = rp[i + 1];
    int    J[MAX_PATCH];
    int    Kpos[MAX_PATCH];
    double Wabs[MAX_PATCH];
    int m_low = 0;
    int max_low = max_patch - 1;
    double a_ii = 0.0;
    bool found_diag = false;

    for (int k = rs; k < re; ++k)
    {
        int col = ci[k];
        double v = (double)aval[k];
        if (col == i)
        {
            a_ii = v;
            found_diag = true;
        }
        else if (col < i)
        {
            double av = fabs(v);
            if (m_low < max_low)
            {
                J[m_low] = col; Kpos[m_low] = k; Wabs[m_low] = av;
                m_low++;
            }
            else if (max_low > 0)
            {
                int min_idx = 0;
                double min_w = Wabs[0];
                for (int t = 1; t < max_low; ++t)
                {
                    if (Wabs[t] < min_w) { min_w = Wabs[t]; min_idx = t; }
                }
                if (av > min_w)
                {
                    J[min_idx] = col; Kpos[min_idx] = k; Wabs[min_idx] = av;
                }
            }
        }
    }
    if (!found_diag)
    {
        if (err_flags) atomicOr(err_flags, kErrMissingDiag);
        return;
    }
    if (!(a_ii > 0.0) || !is_finite_f64(a_ii))
    {
        if (err_flags) atomicOr(err_flags, kErrInvalidDiag);
        return;
    }

    int m = m_low + 1;
    J[m - 1] = i;

    double L[MAX_PATCH * MAX_PATCH];
    bool ok = true;
    for (int a = 0; a < m && ok; ++a)
    {
        for (int b = 0; b <= a; ++b)
        {
            double sum = (a == m - 1 && b == m - 1) ? a_ii : csr_lookup_fast(rp, ci, aval, J[a], J[b]);
            for (int k = 0; k < b; ++k) sum -= L[a * MAX_PATCH + k] * L[b * MAX_PATCH + k];
            if (a == b)
            {
                if (!(sum > 0.0) || !is_finite_f64(sum))
                {
                    ok = false;
                    break;
                }
                L[a * MAX_PATCH + a] = sqrt(sum);
            }
            else
            {
                double val = sum / L[b * MAX_PATCH + b];
                if (!is_finite_f64(val))
                {
                    ok = false;
                    break;
                }
                L[a * MAX_PATCH + b] = val;
            }
        }
    }

    double g[MAX_PATCH] = {};
    if (ok)
    {
        double llast = L[(m - 1) * MAX_PATCH + (m - 1)];
        g[m - 1] = 1.0 / llast;
        if (!is_finite_f64(g[m - 1]))
        {
            ok = false;
        }
        else
        {
            for (int a = m - 2; a >= 0; --a)
            {
                double s = 0.0;
                for (int b = a + 1; b < m; ++b) s += L[b * MAX_PATCH + a] * g[b];
                g[a] = -s / L[a * MAX_PATCH + a];
                if (!is_finite_f64(g[a]))
                {
                    ok = false;
                    break;
                }
            }
        }
    }

    double g_d = 0.0;
    double scale = 0.0;
    bool diagonal_only = !ok || (m == 1);
    if (!ok)
    {
        for (int a = 0; a < m - 1; ++a) g[a] = 0.0;
        g_d = 1.0 / sqrt(a_ii);
        if (!(g_d > 0.0) || !is_finite_f64(g_d))
        {
            if (err_flags) atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        scale = g_d;
        g[m - 1] = g_d;
        if (diag_fallback_count) atomicAdd(diag_fallback_count, 1);
    }
    else
    {
        g_d = g[m - 1];
    }

    if (mode == 2)
    {
        if (!(g_d > 0.0) || !is_finite_f64(g_d))
        {
            if (err_flags) atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        g_diag_f64[i] = g_d;
        for (int a = 0; a < m - 1; ++a)
        {
            int j = J[a];
            int k_low = Kpos[a];
            int k_up = csr_find_pos_fast(rp, ci, j, i);
            if (k_up < 0 && err_flags) atomicOr(err_flags, kErrNonsymmetric);
            double gv = g[a];
            g_f64[k_low] = gv;
            if (k_up >= 0) g_f64[k_up] = gv;
        }
        return;
    }

    double omax = 0.0;
    for (int a = 0; a < m - 1; ++a) omax = fmax(omax, fabs(g[a]));
    if (ok)
    {
        scale = (mode == 3) ? fmax(fabs(g_d), omax) : ((omax > 0.0) ? omax : fabs(g_d));
    }
    if (!(scale != 0.0) || !is_finite_f64(scale))
    {
        if (err_flags) atomicOr(err_flags, kErrUnrepresentable);
        return;
    }

    uint8_t q_code[MAX_PATCH];
    __half  q_f16[MAX_PATCH];
    double  q_dec[MAX_PATCH];
    for (int a = 0; a < m - 1; ++a)
    {
        double u = g[a] / scale;
        if (mode == 0 || mode == 3 || mode == 4)
        {
            uint8_t code = (mode == 0) ? encode_nvfp4(u) : encode_int4(u);
            q_code[a] = code;
            q_dec[a] = (double)decode_nibble(code, mode);
        }
        else
        {
            __half hv = __double2half(u);
            q_f16[a] = hv;
            q_dec[a] = (double)__half2float(hv);
        }
    }

    if (use_reproject && ok && m > 1)
    {
        RowProjStatus status = project_row_device(m, q_dec, L, rp, ci, aval, J, &g_d, &scale);
        if (status == RowProjStatus::InvalidInput)
        {
            if (err_flags) atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        if (status == RowProjStatus::InvalidStorage)
        {
            if (err_flags) atomicOr(err_flags, kErrUnrepresentable);
            return;
        }
        if (status == RowProjStatus::Fallback)
        {
            g_d = 1.0 / sqrt(a_ii);
            if (!(g_d > 0.0) || !is_finite_f64(g_d))
            {
                if (err_flags) atomicOr(err_flags, kErrInvalidDiag);
                return;
            }
            scale = g_d;
            diagonal_only = true;
            if (diag_fallback_count) atomicAdd(diag_fallback_count, 1);
            for (int a = 0; a < m - 1; ++a)
            {
                q_code[a] = (mode == 0) ? 0u : 8u;
                q_f16[a]  = __double2half(0.0);
            }
        }
    }

    if (!is_storable_f32(g_d) || !is_storable_f32(scale))
    {
        if (err_flags) atomicOr(err_flags, kErrUnrepresentable);
        return;
    }
    if (diagonal_only)
    {
        double sd = (double)(float)g_d;
        double se = sd * sd * a_ii;
        if (!is_finite_f64(se) || fabs(se - 1.0) > kEnergyTol)
        {
            if (err_flags) atomicOr(err_flags, kErrUnrepresentable);
            return;
        }
    }

    double ratio = fabs(scale) / g_d;
    if (err_flags && (ratio < 1e-6 || ratio > 1e6 ||
                      fabs(scale) < sqrt((double)FLT_MIN) ||
                      fabs(scale) > sqrt((double)FLT_MAX)))
    {
        atomicOr(err_flags, kFlagWideApply);
    }

    g_diag[i] = (float)g_d;
    g_off_scale[i] = (float)scale;

    for (int a = 0; a < m - 1; ++a)
    {
        int j = J[a];
        int k_low = Kpos[a];
        int k_up = csr_find_pos_fast(rp, ci, j, i);
        if (k_up < 0 && err_flags) atomicOr(err_flags, kErrNonsymmetric);
        if (mode == 0 || mode == 3 || mode == 4)
        {
            q_tmp[k_low] = q_code[a];
            if (k_up >= 0) q_tmp[k_up] = q_code[a];
        }
        else
        {
            g_f16[k_low] = q_f16[a];
            if (k_up >= 0) g_f16[k_up] = q_f16[a];
        }
    }
}

__global__ void pack_nibbles_kernel(int nnz, const uint8_t* __restrict__ q_tmp, uint8_t* __restrict__ q_packed)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = b << 1;
    if (idx >= nnz) return;
    uint8_t lo = q_tmp[idx] & 0x0F;
    uint8_t hi = (idx + 1 < nnz) ? (q_tmp[idx + 1] & 0x0F) : 0u;
    q_packed[b] = (uint8_t)(lo | (hi << 4));
}

// Certified Gershgorin upper bound on lambda_max(G_q A G_q^T):
// Step 1: u = |G_q|^T * 1,  Step 2: v = |A| * u,  Step 3: max_i (|G_q| * v)_i
template <int MODE>
__global__ void gershgorin_abs_gt1_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    double* __restrict__ u)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int rs = rp[j], re = rp[j + 1];
    double s = (MODE == 2) ? fabs(g_diag_f64[j]) : fabs((double)g_diag[j]);
    for (int k = rs; k < re; ++k)
    {
        int i = ci[k];
        if (i > j)
        {
            double w;
            if constexpr (MODE == 2) w = fabs(g_f64[k]);
            else if constexpr (MODE == 1) w = fabs((double)__half2float(g_f16[k])) * fabs((double)g_off_scale[i]);
            else
            {
                uint8_t byte = q_packed[k >> 1];
                uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                w = fabs((double)decode_nibble(nib, MODE)) * fabs((double)g_off_scale[i]);
            }
            s += w;
        }
    }
    u[j] = s;
}

template <typename T>
__global__ void gershgorin_abs_au_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    const double* __restrict__ u, double* __restrict__ v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    for (int k = rp[i]; k < rp[i + 1]; ++k) s += fabs((double)aval[k]) * u[ci[k]];
    v[i] = s;
}

template <int MODE>
__global__ void gershgorin_abs_gv_max_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const double* __restrict__ v, double* __restrict__ out_max)
{
    __shared__ double sh[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    if (i < n)
    {
        int rs = rp[i], re = rp[i + 1];
        s = ((MODE == 2) ? fabs(g_diag_f64[i]) : fabs((double)g_diag[i])) * v[i];
        double sc = (MODE == 2) ? 1.0 : fabs((double)g_off_scale[i]);
        double off = 0.0;
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                double w;
                if constexpr (MODE == 2) w = fabs(g_f64[k]);
                else if constexpr (MODE == 1) w = fabs((double)__half2float(g_f16[k]));
                else
                {
                    uint8_t byte = q_packed[k >> 1];
                    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    w = fabs((double)decode_nibble(nib, MODE));
                }
                off += w * v[col];
            }
        }
        s += sc * off;
    }
    sh[tid] = s;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1)
    {
        if (tid < off) sh[tid] = fmax(sh[tid], sh[tid + off]);
        __syncthreads();
    }
    if (tid == 0)
    {
        unsigned long long* addr = reinterpret_cast<unsigned long long*>(out_max);
        unsigned long long old = *addr, assumed;
        unsigned long long val = __double_as_longlong(sh[0]);
        do
        {
            assumed = old;
            if (__longlong_as_double(assumed) >= sh[0]) break;
            old = atomicCAS(addr, assumed, val);
        } while (assumed != old);
    }
}

// Fused residual r = b - A * x
template <typename TA, typename TB>
__global__ void fused_residual_kernel(int n, const int* __restrict__ rp, const int* __restrict__ ci,
                                      const TA* __restrict__ aval, const TB* __restrict__ b,
                                      const TB* __restrict__ x, TB* __restrict__ r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    TB s = (TB)0.0;
    for (int k = rp[i]; k < rp[i + 1]; ++k) s += (TB)aval[k] * __ldg(&x[ci[k]]);
    r[i] = b[i] - s;
}

// Pass 1 (Zero-Duplicate-Index): stores 4 B/row scale-factored y_i = s_i t_i for ordinary FP32,
// or 8 B/row unscaled FP64 t_i when WIDE is true (FP64 vectors or extreme row scales).
template <int MODE, bool WIDE, typename TB>
__global__ void bitfsai_fwd_kernel(
    int n,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const TB* __restrict__ r,
    float* __restrict__ y,
    double* __restrict__ t_f64)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int rs = rp[i], re = rp[i + 1];

    if constexpr (MODE == 2)
    {
        double acc = g_diag_f64[i] * (double)r[i];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i) acc += g_f64[k] * (double)__ldg(&r[col]);
        }
        t_f64[i] = acc;
    }
    else if constexpr (WIDE)
    {
        double sc = (double)g_off_scale[i];
        double acc = (double)g_diag[i] * (double)r[i];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                float w;
                if constexpr (MODE == 0 || MODE == 3 || MODE == 4)
                {
                    uint8_t byte = q_packed[k >> 1];
                    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    w = decode_nibble(nib, MODE);
                }
                else
                {
                    w = __half2float(g_f16[k]);
                }
                acc = fma(sc * (double)w, (double)__ldg(&r[col]), acc);
            }
        }
        t_f64[i] = acc;
    }
    else
    {
        float sc = g_off_scale[i];
        float acc = 0.0f;
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                float w;
                if constexpr (MODE == 0 || MODE == 3 || MODE == 4)
                {
                    uint8_t byte = q_packed[k >> 1];
                    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    w = decode_nibble(nib, MODE);
                }
                else
                {
                    w = __half2float(g_f16[k]);
                }
                acc = fmaf(w, (float)__ldg(&r[col]), acc);
            }
        }
        float ti = fmaf(sc, acc, g_diag[i] * (float)r[i]);
        y[i] = sc * ti;
    }
}

// Pass 2 (Zero-Duplicate-Index): x[j] += omega * (G_q^T * t)_j
template <int MODE, bool WIDE, bool X_IS_ZERO, typename TB>
__global__ void bitfsai_tr_kernel(
    int n, double omega,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const float* __restrict__ y,
    const double* __restrict__ t_f64,
    TB* __restrict__ x)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int rs = rp[j], re = rp[j + 1];

    if constexpr (MODE == 2)
    {
        double acc = g_diag_f64[j] * t_f64[j];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j) acc += g_f64[k] * __ldg(&t_f64[col]);
        }
        if constexpr (X_IS_ZERO) x[j] = (TB)(omega * acc);
        else x[j] += (TB)(omega * acc);
    }
    else if constexpr (WIDE)
    {
        double acc = (double)g_diag[j] * t_f64[j];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j)
            {
                float w;
                if constexpr (MODE == 0 || MODE == 3 || MODE == 4)
                {
                    uint8_t byte = q_packed[k >> 1];
                    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    w = decode_nibble(nib, MODE);
                }
                else
                {
                    w = __half2float(g_f16[k]);
                }
                acc = fma((double)__ldg(&g_off_scale[col]) * (double)w, __ldg(&t_f64[col]), acc);
            }
        }
        if constexpr (X_IS_ZERO) x[j] = (TB)(omega * acc);
        else x[j] += (TB)(omega * acc);
    }
    else
    {
        float acc = g_diag[j] * (y[j] / g_off_scale[j]);
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j)
            {
                float w;
                if constexpr (MODE == 0 || MODE == 3 || MODE == 4)
                {
                    uint8_t byte = q_packed[k >> 1];
                    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    w = decode_nibble(nib, MODE);
                }
                else
                {
                    w = __half2float(g_f16[k]);
                }
                acc = fmaf(w, __ldg(&y[col]), acc);
            }
        }
        if constexpr (X_IS_ZERO) x[j] = (TB)(omega * (double)acc);
        else x[j] += (TB)(omega * (double)acc);
    }
}

} // namespace bitfsai_detail

template <class T_Config>
BitFSAISolver_Base<T_Config>::BitFSAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope)
{
    std::string mode_str = cfg.AMG_Config::template getParameter<std::string>("bitfsai_mode", cfg_scope);
    if      (mode_str == "nvfp4")      m_mode = 0;
    else if (mode_str == "fp16")       m_mode = 1;
    else if (mode_str == "fp64")       m_mode = 2;
    else if (mode_str == "int4")       m_mode = 3;
    else if (mode_str == "int4_split") m_mode = 4;
    else if (mode_str == "auto")       m_mode = 5;
    else FatalError("Unknown bitfsai_mode value", AMGX_ERR_CONFIGURATION);
    m_active_mode = (m_mode == 5) ? 4 : m_mode;

    m_omega_cfg = cfg.AMG_Config::template getParameter<double>("bitfsai_omega", cfg_scope);
    if (m_omega_cfg <= 0.0)
    {
        FatalError("bitfsai_omega must be strictly positive", AMGX_ERR_CONFIGURATION);
    }
    m_omega = m_omega_cfg;

    m_max_patch_cfg = cfg.AMG_Config::template getParameter<int>("bitfsai_max_patch", cfg_scope);
    if (m_max_patch_cfg < 0 || m_max_patch_cfg > bitfsai_detail::MAX_PATCH)
    {
        FatalError("bitfsai_max_patch must be in [0, 16]", AMGX_ERR_CONFIGURATION);
    }
    m_use_safe_omega = (cfg.AMG_Config::template getParameter<int>("bitfsai_safe_omega", cfg_scope) != 0);
    m_free_scratch   = (cfg.AMG_Config::template getParameter<int>("bitfsai_free_scratch", cfg_scope) != 0);
    m_use_reproject  = (cfg.AMG_Config::template getParameter<int>("bitfsai_reproject", cfg_scope) != 0);
}

template <class T_Config>
void BitFSAISolver_Base<T_Config>::free_buffers()
{
    if (d_g_diag)      cudaFree(d_g_diag);
    if (d_g_off_scale) cudaFree(d_g_off_scale);
    if (d_q_packed)    cudaFree(d_q_packed);
    if (d_g_f16)       cudaFree(d_g_f16);
    if (d_g_diag_f64)  cudaFree(d_g_diag_f64);
    if (d_g_f64)       cudaFree(d_g_f64);
    if (d_t_f64)       cudaFree(d_t_f64);
    if (d_y)           cudaFree(d_y);
    if (d_res)         cudaFree(d_res);
    d_g_diag = nullptr;
    d_g_off_scale = nullptr;
    d_q_packed = nullptr;
    d_g_f16 = nullptr;
    d_g_diag_f64 = nullptr;
    d_g_f64 = nullptr;
    d_t_f64 = nullptr;
    d_y = nullptr;
    d_res = nullptr;
    m_n = 0;
    m_nnz = 0;
    m_alloc_mode = -1;
}

template <class T_Config>
BitFSAISolver_Base<T_Config>::~BitFSAISolver_Base()
{
    free_buffers();
}

void release_bitfsai_staging_pool()
{
    bitfsai_detail::g_staging_pool.release_all();
}

template <class T_Config>
void BitFSAISolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    this->m_is_solver_setup = false;
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(this->m_A);
    if (!m_explicit_A)
    {
        this->free_buffers();
        FatalError("BitFSAISolver requires explicit CSR matrix", AMGX_ERR_INTERNAL);
    }
    setup_device(*m_explicit_A, reuse_matrix_structure);
}

template <class T_Config>
AMGX_STATUS BitFSAISolver_Base<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    if (xIsZero) x.dirtybit = 0;
    ViewType oldView = m_explicit_A->currentView();
    m_explicit_A->setViewExterior();
    smooth_device(*m_explicit_A, b, x, xIsZero);
    x.dirtybit = 1;
    m_explicit_A->setView(oldView);
    return this->converged(b, x);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BitFSAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setup_device(
    const Matrix_d &A, bool reuse)
{
    typedef typename Matrix_d::value_type ValueTypeA;
    typedef typename TConfig_d::VecPrec   ValueTypeB;
    if constexpr (!((std::is_same_v<ValueTypeA, double> || std::is_same_v<ValueTypeA, float>) &&
                    (std::is_same_v<ValueTypeB, double> || std::is_same_v<ValueTypeB, float>)))
    {
        FatalError("BitFSAISolver supports only real float/double matrix and vector modes (dDDI, dFFI, dDFI)",
                   AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        if (!A.is_matrix_singleGPU())
        {
            this->free_buffers();
            FatalError("BitFSAISolver currently supports single-GPU matrices only", AMGX_ERR_NOT_IMPLEMENTED);
        }
        if (!A.hasProps(CSR))
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires CSR matrix storage", AMGX_ERR_NOT_IMPLEMENTED);
        }
        if (A.hasProps(DIAG))
        {
            this->free_buffers();
            FatalError("BitFSAISolver does not support separate DIAG storage", AMGX_ERR_NOT_IMPLEMENTED);
        }
        if (A.get_block_dimx() != 1 || A.get_block_dimy() != 1)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires scalar (1x1 block) CSR matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }
        if (A.get_num_rows() != A.get_num_cols())
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires square CSR matrices", AMGX_ERR_BAD_PARAMETERS);
        }
        int n = (int)A.get_num_rows();
        int nnz = (int)A.get_num_nz();
        if (n <= 0)
        {
            this->free_buffers();
            return;
        }
        if (nnz <= 0)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires explicit in-row diagonal entries in CSR", AMGX_ERR_BAD_PARAMETERS);
        }

        auto check_cuda = [&](cudaError_t err, const char* msg)
        {
            if (err != cudaSuccess)
            {
                this->free_buffers();
                FatalError(msg, AMGX_ERR_CUDA_FAILURE);
            }
        };

        int avg_deg = nnz / std::max(n, 1);
        int max_patch = (this->m_max_patch_cfg > 0) ? this->m_max_patch_cfg : ((avg_deg > 22) ? 8 : 12);
        int th = 256, bl = (int)(((size_t)n + (size_t)th - 1) / (size_t)th);

        int eff_mode = this->m_mode;
        if (this->m_mode == 5)
        {
            double* d_errs = nullptr;
            check_cuda(cudaMalloc(&d_errs, 2 * sizeof(double)), "BitFSAISolver failed to allocate auto-probe buffer");
            cudaError_t err = cudaMemset(d_errs, 0, 2 * sizeof(double));
            if (err == cudaSuccess)
            {
                bitfsai_detail::bitfsai_energy_probe_kernel<<<bl, th>>>(
                    n, max_patch, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), d_errs);
                err = cudaGetLastError();
            }
            double h_errs[2] = {0.0, 0.0};
            if (err == cudaSuccess)
            {
                err = cudaMemcpy(h_errs, d_errs, 2 * sizeof(double), cudaMemcpyDeviceToHost);
            }
            cudaFree(d_errs);
            check_cuda(err, "BitFSAISolver auto-mode energy probe failed");
            eff_mode = (h_errs[1] < h_errs[0]) ? 0 : 4;
        }
        this->m_active_mode = eff_mode;

        const bool uses_4bit = (eff_mode == 0 || eff_mode == 3 || eff_mode == 4);
        const bool alloc_4bit = (this->m_alloc_mode == 0 || this->m_alloc_mode == 3 || this->m_alloc_mode == 4);
        const bool layout_compat = (eff_mode == this->m_alloc_mode) || (uses_4bit && alloc_4bit);
        if (!reuse || n != this->m_n || nnz != this->m_nnz || !layout_compat)
        {
            this->free_buffers();
            check_cuda(cudaMalloc(&this->d_res, (size_t)n * sizeof(ValueTypeB)),
                       "BitFSAISolver failed to allocate residual buffer");
            if (eff_mode == 2)
            {
                check_cuda(cudaMalloc(&this->d_g_diag_f64, (size_t)n * sizeof(double)),
                           "BitFSAISolver failed to allocate FP64 diagonal buffer");
                check_cuda(cudaMalloc(&this->d_g_f64, (size_t)nnz * sizeof(double)),
                           "BitFSAISolver failed to allocate FP64 factor buffer");
            }
            else
            {
                check_cuda(cudaMalloc(&this->d_g_diag, (size_t)n * sizeof(float)),
                           "BitFSAISolver failed to allocate FP32 diagonal buffer");
                check_cuda(cudaMalloc(&this->d_g_off_scale, (size_t)n * sizeof(float)),
                           "BitFSAISolver failed to allocate FP32 scale buffer");
                if (eff_mode == 1)
                {
                    check_cuda(cudaMalloc(&this->d_g_f16, (size_t)nnz * sizeof(__half)),
                               "BitFSAISolver failed to allocate FP16 factor buffer");
                }
                else
                {
                    check_cuda(cudaMalloc(&this->d_q_packed, ((size_t)nnz + 1) / 2),
                               "BitFSAISolver failed to allocate 4-bit packed factor buffer");
                }
            }
        }
        else
        {
            // Invalidate reusable dimensions until setup validation completes.
            this->m_n = 0;
            this->m_nnz = 0;
            this->m_alloc_mode = -1;
        }

        // Validate CSR structure, diagonal positivity, FP32 storability, and wide-apply threshold
        int* d_err_flags = nullptr;
        check_cuda(cudaMalloc(&d_err_flags, sizeof(int)), "BitFSAISolver failed to allocate error flag buffer");
        cudaError_t build_err = cudaMemset(d_err_flags, 0, sizeof(int));
        if (build_err != cudaSuccess)
        {
            cudaFree(d_err_flags);
            check_cuda(build_err, "BitFSAISolver failed to initialize error flag buffer");
        }

        if (uses_4bit)
        {
            auto lease = bitfsai_detail::g_staging_pool.acquire((size_t)nnz, this->m_free_scratch);
            uint8_t* d_q_tmp = lease.data();
            if (!d_q_tmp)
            {
                cudaFree(d_err_flags);
                this->free_buffers();
                FatalError("BitFSAISolver failed to allocate device staging workspace", AMGX_ERR_INTERNAL);
            }
            build_err = cudaMemset(d_q_tmp, (eff_mode == 0) ? 0 : 8, (size_t)nnz);
            if (build_err == cudaSuccess)
            {
                bitfsai_detail::bitfsai_build_kernel<<<bl, th>>>(
                    n, eff_mode, max_patch, this->m_use_reproject,
                    A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    d_q_tmp, this->d_g_f16, this->d_g_f64, d_err_flags, nullptr);
                size_t nbytes = ((size_t)nnz + 1) / 2;
                int pack_bl = (int)((nbytes + (size_t)th - 1) / (size_t)th);
                bitfsai_detail::pack_nibbles_kernel<<<pack_bl, th>>>(nnz, d_q_tmp, this->d_q_packed);
                build_err = cudaGetLastError();
            }
            lease.finish_and_unlock(0);
        }
        else
        {
            if (eff_mode == 1) build_err = cudaMemset(this->d_g_f16, 0, (size_t)nnz * sizeof(__half));
            else               build_err = cudaMemset(this->d_g_f64, 0, (size_t)nnz * sizeof(double));
            if (build_err == cudaSuccess)
            {
                bitfsai_detail::bitfsai_build_kernel<<<bl, th>>>(
                    n, eff_mode, max_patch, this->m_use_reproject,
                    A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    nullptr, this->d_g_f16, this->d_g_f64, d_err_flags, nullptr);
                build_err = cudaGetLastError();
            }
        }

        int h_err = 0;
        if (build_err == cudaSuccess)
        {
            build_err = cudaMemcpy(&h_err, d_err_flags, sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaFree(d_err_flags);
        check_cuda(build_err, "BitFSAISolver build kernel failed");

        if (h_err & bitfsai_detail::kErrMissingDiag)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires explicit in-row diagonal entries in CSR", AMGX_ERR_BAD_PARAMETERS);
        }
        if (h_err & bitfsai_detail::kErrNonsymmetric)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires structurally symmetric (mirrored) CSR matrices", AMGX_ERR_BAD_PARAMETERS);
        }
        if (h_err & bitfsai_detail::kErrInvalidDiag)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires finite strictly positive diagonals", AMGX_ERR_BAD_PARAMETERS);
        }
        if (h_err & bitfsai_detail::kErrUnrepresentable)
        {
            this->free_buffers();
            FatalError("BitFSAISolver factor cannot be represented in FP32", AMGX_ERR_BAD_PARAMETERS);
        }

        this->m_wide_apply = (eff_mode == 2) || std::is_same_v<ValueTypeB, double> ||
                             ((h_err & bitfsai_detail::kFlagWideApply) != 0);
        if (this->m_wide_apply)
        {
            if (this->d_y) { cudaFree(this->d_y); this->d_y = nullptr; }
            if (!this->d_t_f64)
            {
                check_cuda(cudaMalloc(&this->d_t_f64, (size_t)n * sizeof(double)),
                           "BitFSAISolver failed to allocate FP64 intermediate buffer");
            }
        }
        else
        {
            if (this->d_t_f64) { cudaFree(this->d_t_f64); this->d_t_f64 = nullptr; }
            if (!this->d_y)
            {
                check_cuda(cudaMalloc(&this->d_y, (size_t)n * sizeof(float)),
                           "BitFSAISolver failed to allocate FP32 intermediate buffer");
            }
        }

        this->m_omega = this->m_omega_cfg;
        if (this->m_use_safe_omega)
        {
            double *d_u = nullptr, *d_v = nullptr, *d_max = nullptr;
            cudaError_t gerr = cudaMalloc(&d_u, (size_t)n * sizeof(double));
            if (gerr == cudaSuccess) gerr = cudaMalloc(&d_v, (size_t)n * sizeof(double));
            if (gerr == cudaSuccess) gerr = cudaMalloc(&d_max, sizeof(double));
            if (gerr == cudaSuccess) gerr = cudaMemset(d_max, 0, sizeof(double));
            if (gerr == cudaSuccess)
            {
                auto run_gersh = [&](auto mode_tag)
                {
                    constexpr int M = decltype(mode_tag)::value;
                    bitfsai_detail::gershgorin_abs_gt1_kernel<M><<<bl, th>>>(
                        n, A.row_offsets.raw(), A.col_indices.raw(),
                        this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                        this->d_q_packed, this->d_g_f16, this->d_g_f64, d_u);
                    bitfsai_detail::gershgorin_abs_au_kernel<<<bl, th>>>(
                        n, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), d_u, d_v);
                    bitfsai_detail::gershgorin_abs_gv_max_kernel<M><<<bl, th>>>(
                        n, A.row_offsets.raw(), A.col_indices.raw(),
                        this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                        this->d_q_packed, this->d_g_f16, this->d_g_f64, d_v, d_max);
                };
                if      (eff_mode == 0) run_gersh(std::integral_constant<int, 0>{});
                else if (eff_mode == 1) run_gersh(std::integral_constant<int, 1>{});
                else if (eff_mode == 2) run_gersh(std::integral_constant<int, 2>{});
                else if (eff_mode == 3) run_gersh(std::integral_constant<int, 3>{});
                else                    run_gersh(std::integral_constant<int, 4>{});
                gerr = cudaGetLastError();
            }
            double h_ub = 0.0;
            if (gerr == cudaSuccess) gerr = cudaMemcpy(&h_ub, d_max, sizeof(double), cudaMemcpyDeviceToHost);
            if (d_u) cudaFree(d_u);
            if (d_v) cudaFree(d_v);
            if (d_max) cudaFree(d_max);
            check_cuda(gerr, "BitFSAISolver safe-omega estimation failed");
            if (h_ub > 0.0) this->m_omega = std::min(this->m_omega_cfg, 1.8 / h_ub);
        }
        this->m_n = n;
        this->m_nnz = nnz;
        this->m_alloc_mode = eff_mode;
        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void BitFSAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smooth_device(
    const Matrix_d &A, const VVector &b, VVector &x, bool xIsZero)
{
    typedef typename Matrix_d::value_type ValueTypeA;
    typedef typename TConfig_d::VecPrec   ValueTypeB;
    if constexpr (!((std::is_same_v<ValueTypeA, double> || std::is_same_v<ValueTypeA, float>) &&
                    (std::is_same_v<ValueTypeB, double> || std::is_same_v<ValueTypeB, float>)))
    {
        FatalError("BitFSAISolver supports only real float/double matrix and vector modes (dDDI, dFFI, dDFI)",
                   AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int n = (int)A.get_num_rows();
        if (n <= 0) return;
        if (this->m_n != n || this->m_alloc_mode < 0 || !this->d_res ||
            (this->m_wide_apply ? !this->d_t_f64 : !this->d_y))
        {
            FatalError("BitFSAISolver is not initialized; setup must succeed before solve", AMGX_ERR_BAD_PARAMETERS);
        }
        int th = 256, bl = (int)(((size_t)n + (size_t)th - 1) / (size_t)th);
        const ValueTypeB* r_ptr = b.raw();
        if (!xIsZero)
        {
            bitfsai_detail::fused_residual_kernel<<<bl, th>>>(
                n, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                b.raw(), x.raw(), this->d_res);
            r_ptr = this->d_res;
        }
        double om = this->m_omega;
        int eff_mode = this->m_active_mode;
        bool wide = this->m_wide_apply;
        auto dispatch_wide = [&](auto mode_tag, auto wide_tag)
        {
            constexpr int M = decltype(mode_tag)::value;
            constexpr bool W = decltype(wide_tag)::value;
            bitfsai_detail::bitfsai_fwd_kernel<M, W><<<bl, th>>>(
                n, A.row_offsets.raw(), A.col_indices.raw(),
                this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                this->d_q_packed, this->d_g_f16, this->d_g_f64,
                r_ptr, this->d_y, this->d_t_f64);
            if (xIsZero)
            {
                bitfsai_detail::bitfsai_tr_kernel<M, W, true><<<bl, th>>>(
                    n, om, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    this->d_y, this->d_t_f64, x.raw());
            }
            else
            {
                bitfsai_detail::bitfsai_tr_kernel<M, W, false><<<bl, th>>>(
                    n, om, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    this->d_y, this->d_t_f64, x.raw());
            }
        };
        auto dispatch = [&](auto mode_tag)
        {
            if (wide) dispatch_wide(mode_tag, std::integral_constant<bool, true>{});
            else      dispatch_wide(mode_tag, std::integral_constant<bool, false>{});
        };
        if      (eff_mode == 0) dispatch(std::integral_constant<int, 0>{});
        else if (eff_mode == 1) dispatch(std::integral_constant<int, 1>{});
        else if (eff_mode == 2) dispatch(std::integral_constant<int, 2>{});
        else if (eff_mode == 3) dispatch(std::integral_constant<int, 3>{});
        else                    dispatch(std::integral_constant<int, 4>{});
        cudaCheckError();
    }
}

#define AMGX_CASE_LINE(CASE) template class BitFSAISolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class BitFSAISolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
