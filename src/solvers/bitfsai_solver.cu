// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause

#include <solvers/bitfsai_solver.h>
#include <basic_types.h>
#include <cutil.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace amgx
{

namespace bitfsai_detail
{

constexpr int MAX_PATCH = 16;

constexpr int kErrInvalidCsr       = 1;
constexpr int kErrMissingDiag      = 2;
constexpr int kErrNonsymmetric     = 4;
constexpr int kErrInvalidDiag      = 8;
constexpr int kErrInvalidValue     = 16;
constexpr int kErrUnrepresentable  = 32;
constexpr int kFlagWideApply       = 64;
constexpr double kEnergyTol        = 1.0e-4;

enum class Mode : int
{
    Int4Split = 0,
    Fp4E2m1   = 1,
    Fp16      = 2,
    Fp64      = 3
};

// Exact FP4 (E2M1) magnitude lattice normalized by 6.0 (bit 3 = sign)
__device__ __constant__ float c_fp4_e2m1_lut[16] = {
    0.0f / 6.0f,  0.5f / 6.0f,  1.0f / 6.0f,  1.5f / 6.0f,
    2.0f / 6.0f,  3.0f / 6.0f,  4.0f / 6.0f,  6.0f / 6.0f,
   -0.0f / 6.0f, -0.5f / 6.0f, -1.0f / 6.0f, -1.5f / 6.0f,
   -2.0f / 6.0f, -3.0f / 6.0f, -4.0f / 6.0f, -6.0f / 6.0f
};

__device__ __forceinline__ bool is_finite_f64(double v)
{
    return v == v && v >= -DBL_MAX && v <= DBL_MAX;
}

__device__ __forceinline__ bool is_storable_f32(double v)
{
    float s = static_cast<float>(v);
    return is_finite_f64(v) && s == s && s != 0.0f && s >= -FLT_MAX && s <= FLT_MAX;
}

__device__ __forceinline__ uint8_t encode_fp4_e2m1(double r)
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
    return (mag == 0u) ? 0u : static_cast<uint8_t>(sgn | mag);
}

__device__ __forceinline__ uint8_t encode_int4(double r)
{
    int q = static_cast<int>(llround(r * 7.0));
    if (q > 7) q = 7;
    if (q < -7) q = -7;
    return static_cast<uint8_t>(q + 8); // [1..15], 8 encodes exact 0
}

__device__ __forceinline__ float decode_nibble(uint8_t nib, Mode mode)
{
    if (mode == Mode::Fp4E2m1) return c_fp4_e2m1_lut[nib & 0x0Fu];
    return static_cast<float>(static_cast<int>(nib & 0x0Fu) - 8) * (1.0f / 7.0f);
}

template <Mode M>
__device__ __forceinline__ float off_weight(int k, const uint8_t* __restrict__ packed,
                                            const __half* __restrict__ f16)
{
    if constexpr (M == Mode::Fp16) return __half2float(f16[k]);
    uint8_t byte = packed[k >> 1];
    uint8_t nib = (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0Fu);
    return decode_nibble(nib, M);
}

template <typename T>
__device__ __forceinline__ double csr_lookup(const int* __restrict__ rp,
                                             const int* __restrict__ ci,
                                             const T* __restrict__ val,
                                             int row, int col)
{
    for (int k = rp[row], re = rp[row + 1]; k < re; ++k)
    {
        if (ci[k] == col) return static_cast<double>(val[k]);
    }
    return 0.0;
}

__device__ __forceinline__ int csr_find_pos(const int* __restrict__ rp,
                                            const int* __restrict__ ci,
                                            int row, int col)
{
    for (int k = rp[row], re = rp[row + 1]; k < re; ++k)
    {
        if (ci[k] == col) return k;
    }
    return -1;
}

__global__ void validate_csr_rows_kernel(int n, int nnz,
                                         const int* __restrict__ rp,
                                         int* __restrict__ err_flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
    {
        if (rp[0] != 0 || rp[n] != nnz) atomicOr(err_flags, kErrInvalidCsr);
    }
    if (i >= n) return;
    int rs = rp[i], re = rp[i + 1];
    if (rs < 0 || re < rs || re > nnz) atomicOr(err_flags, kErrInvalidCsr);
}

__global__ void validate_csr_cols_kernel(int n,
                                         const int* __restrict__ rp,
                                         const int* __restrict__ ci,
                                         const int* __restrict__ row_flags,
                                         int* __restrict__ col_flags)
{
    if (*row_flags & kErrInvalidCsr) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int rs = rp[i], re = rp[i + 1];
    bool strictly_sorted = true;
    int prev = -1;
    for (int k = rs; k < re; ++k)
    {
        int c = ci[k];
        if (c < 0 || c >= n)
        {
            atomicOr(col_flags, kErrInvalidCsr);
            return;
        }
        if (k > rs && c <= prev) strictly_sorted = false;
        prev = c;
    }
    if (!strictly_sorted)
    {
        for (int a = rs; a < re; ++a)
        {
            int ca = ci[a];
            for (int b = a + 1; b < re; ++b)
            {
                if (ci[b] == ca)
                {
                    atomicOr(col_flags, kErrInvalidCsr);
                    return;
                }
            }
        }
    }
}

enum class RowProjStatus : uint8_t
{
    Applied,
    Fallback,
    InvalidInput,
    InvalidStorage
};

// Orthogonal-component 2x2 A-energy re-projection with Kahan-compensated stored-FP32 energy check.
template <typename T>
__device__ __forceinline__ RowProjStatus project_row_device(
    int m, const double* q_dec, const double* L,
    const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    const int* J, double* diag_out, double* scale_out)
{
    if (m < 2) return RowProjStatus::Fallback;
    double c22 = csr_lookup(rp, ci, aval, J[m - 1], J[m - 1]);
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

    float g_f32 = static_cast<float>(g_opt);
    float s_f32 = static_cast<float>(s_opt);
    if (s_f32 == 0.0f) return RowProjStatus::Fallback;
    if (!is_storable_f32(g_opt) || !is_storable_f32(s_opt)) return RowProjStatus::InvalidStorage;

    double sg = static_cast<double>(g_f32);
    double ss = static_cast<double>(s_f32);
    double energy = 0.0, comp = 0.0, sum_abs = 0.0;
    for (int a = 0; a < m; ++a)
    {
        double wa = (a == m - 1) ? sg : ss * q_dec[a];
        for (int b = 0; b < m; ++b)
        {
            double wb = (b == m - 1) ? sg : ss * q_dec[b];
            double term = wa * csr_lookup(rp, ci, aval, J[a], J[b]) * wb;
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

// GPU-parallel Zero-Duplicate-Index FSAI(1) Numeric Setup Kernel.
// Local principal-submatrix Cholesky always executes in FP64 regardless of V-cycle precision.
template <typename T>
__global__ void bitfsai_build_kernel(
    int n, Mode mode, int max_patch, bool use_reproject, bool check_full_sym,
    const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    float* __restrict__ g_diag,
    float* __restrict__ g_off_scale,
    double* __restrict__ g_diag_f64,
    uint8_t* __restrict__ q_tmp,
    __half* __restrict__ g_f16,
    double* __restrict__ g_f64,
    int* __restrict__ err_flags)
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
    bool nonsymmetric = false;
    bool invalid_val = false;

    for (int k = rs; k < re; ++k)
    {
        int col = ci[k];
        double v = static_cast<double>(aval[k]);
        if (!is_finite_f64(v)) invalid_val = true;
        if (col == i)
        {
            a_ii = v;
            found_diag = true;
        }
        else
        {
            if (check_full_sym && csr_find_pos(rp, ci, col, i) < 0) nonsymmetric = true;
            if (col < i)
            {
                double av = fabs(v);
                if (m_low < max_low)
                {
                    J[m_low] = col; Kpos[m_low] = k; Wabs[m_low] = av;
                    ++m_low;
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
    }
    if (!found_diag)
    {
        atomicOr(err_flags, kErrMissingDiag);
        return;
    }
    if (nonsymmetric)
    {
        atomicOr(err_flags, kErrNonsymmetric);
        return;
    }
    if (!(a_ii > 0.0) || !is_finite_f64(a_ii))
    {
        atomicOr(err_flags, kErrInvalidDiag);
        return;
    }
    if (invalid_val)
    {
        atomicOr(err_flags, kErrInvalidValue);
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
            double sum = (a == m - 1 && b == m - 1) ? a_ii : csr_lookup(rp, ci, aval, J[a], J[b]);
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
            atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        scale = g_d;
        g[m - 1] = g_d;
    }
    else
    {
        g_d = g[m - 1];
    }

    if (mode == Mode::Fp64)
    {
        if (!(g_d > 0.0) || !is_finite_f64(g_d))
        {
            atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        g_diag_f64[i] = g_d;
        for (int a = 0; a < m - 1; ++a)
        {
            int k_low = Kpos[a];
            int k_up = csr_find_pos(rp, ci, J[a], i);
            g_f64[k_low] = g[a];
            if (k_up >= 0) g_f64[k_up] = g[a];
        }
        return;
    }

    double omax = 0.0;
    for (int a = 0; a < m - 1; ++a) omax = fmax(omax, fabs(g[a]));
    if (ok)
    {
        scale = (omax > 0.0) ? omax : fabs(g_d);
    }
    if (!(scale != 0.0) || !is_finite_f64(scale))
    {
        atomicOr(err_flags, kErrUnrepresentable);
        return;
    }

    const bool is_4bit = (mode == Mode::Int4Split || mode == Mode::Fp4E2m1);
    uint8_t q_code[MAX_PATCH];
    __half  q_f16[MAX_PATCH];
    double  q_dec[MAX_PATCH];
    for (int a = 0; a < m - 1; ++a)
    {
        double u = g[a] / scale;
        if (is_4bit)
        {
            uint8_t code = (mode == Mode::Fp4E2m1) ? encode_fp4_e2m1(u) : encode_int4(u);
            q_code[a] = code;
            q_dec[a] = static_cast<double>(decode_nibble(code, mode));
        }
        else
        {
            __half hv = __double2half(u);
            q_f16[a] = hv;
            q_dec[a] = static_cast<double>(__half2float(hv));
        }
    }

    if (use_reproject && ok && m > 1)
    {
        RowProjStatus status = project_row_device(m, q_dec, L, rp, ci, aval, J, &g_d, &scale);
        if (status == RowProjStatus::InvalidInput)
        {
            atomicOr(err_flags, kErrInvalidDiag);
            return;
        }
        if (status == RowProjStatus::InvalidStorage)
        {
            atomicOr(err_flags, kErrUnrepresentable);
            return;
        }
        if (status == RowProjStatus::Fallback)
        {
            g_d = 1.0 / sqrt(a_ii);
            if (!(g_d > 0.0) || !is_finite_f64(g_d))
            {
                atomicOr(err_flags, kErrInvalidDiag);
                return;
            }
            scale = g_d;
            diagonal_only = true;
            for (int a = 0; a < m - 1; ++a)
            {
                q_code[a] = (mode == Mode::Fp4E2m1) ? 0u : 8u;
                q_f16[a]  = __double2half(0.0);
            }
        }
    }

    if (!is_storable_f32(g_d) || !is_storable_f32(scale))
    {
        atomicOr(err_flags, kErrUnrepresentable);
        return;
    }
    if (diagonal_only)
    {
        double sd = static_cast<double>(static_cast<float>(g_d));
        double se = sd * sd * a_ii;
        if (!is_finite_f64(se) || fabs(se - 1.0) > kEnergyTol)
        {
            atomicOr(err_flags, kErrUnrepresentable);
            return;
        }
    }

    double ratio = fabs(scale) / g_d;
    if (ratio < 1e-6 || ratio > 1e6 ||
        fabs(scale) < sqrt(static_cast<double>(FLT_MIN)) ||
        fabs(scale) > sqrt(static_cast<double>(FLT_MAX)))
    {
        atomicOr(err_flags, kFlagWideApply);
    }

    g_diag[i] = static_cast<float>(g_d);
    g_off_scale[i] = static_cast<float>(scale);

    for (int a = 0; a < m - 1; ++a)
    {
        int k_low = Kpos[a];
        int k_up = csr_find_pos(rp, ci, J[a], i);
        if (is_4bit)
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

__global__ void pack_nibbles_kernel(int nnz, const uint8_t* __restrict__ q_tmp,
                                    uint8_t* __restrict__ q_packed)
{
    uint64_t b = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= (static_cast<uint64_t>(nnz) + 1ULL) / 2ULL) return;
    int idx = static_cast<int>(b << 1);
    uint8_t lo = q_tmp[idx] & 0x0Fu;
    uint8_t hi = (idx + 1 < nnz) ? static_cast<uint8_t>(q_tmp[idx + 1] & 0x0Fu) : 0u;
    q_packed[b] = static_cast<uint8_t>(lo | (hi << 4));
}

// Conservative IEEE-754 upward-rounded Gershgorin bound U_upper >= || |G_q| |A| |G_q|^T 1 ||_inf.
// Step 1: u = |G_q|^T * 1, Step 2: v = |A| * u, Step 3: max_i (|G_q| * v)_i.
template <Mode M>
__global__ void gershgorin_abs_gt1_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    double* __restrict__ u, int* __restrict__ overflow_flag)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int rs = rp[j], re = rp[j + 1];
    double s = (M == Mode::Fp64) ? fabs(g_diag_f64[j]) : fabs(static_cast<double>(g_diag[j]));
    for (int k = rs; k < re; ++k)
    {
        int i = ci[k];
        if (i > j)
        {
            double w;
            if constexpr (M == Mode::Fp64)
            {
                w = fabs(g_f64[k]);
            }
            else
            {
                double aw = fabs(static_cast<double>(off_weight<M>(k, q_packed, g_f16)));
                double asc = fabs(static_cast<double>(g_off_scale[i]));
                w = __dmul_ru(asc, aw);
            }
            s = __dadd_ru(s, w);
        }
    }
    if (!is_finite_f64(s)) atomicOr(overflow_flag, 1);
    u[j] = s;
}

template <typename T>
__global__ void gershgorin_abs_au_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci, const T* __restrict__ aval,
    const double* __restrict__ u, double* __restrict__ v, int* __restrict__ overflow_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    for (int k = rp[i]; k < rp[i + 1]; ++k)
    {
        double term = __dmul_ru(fabs(static_cast<double>(aval[k])), u[ci[k]]);
        s = __dadd_ru(s, term);
    }
    if (!is_finite_f64(s)) atomicOr(overflow_flag, 1);
    v[i] = s;
}

template <Mode M>
__global__ void gershgorin_abs_gv_max_kernel(
    int n, const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const double* __restrict__ v, double* __restrict__ out_max, int* __restrict__ overflow_flag)
{
    __shared__ double sh[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    double s = 0.0;
    if (i < n)
    {
        int rs = rp[i], re = rp[i + 1];
        double dg = (M == Mode::Fp64) ? fabs(g_diag_f64[i]) : fabs(static_cast<double>(g_diag[i]));
        s = __dmul_ru(dg, v[i]);
        double sc = (M == Mode::Fp64) ? 1.0 : fabs(static_cast<double>(g_off_scale[i]));
        double off = 0.0;
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                double aw = (M == Mode::Fp64)
                                ? fabs(g_f64[k])
                                : fabs(static_cast<double>(off_weight<M>(k, q_packed, g_f16)));
                double term = __dmul_ru(aw, v[col]);
                off = __dadd_ru(off, term);
            }
        }
        s = __dadd_ru(s, __dmul_ru(sc, off));
        if (!is_finite_f64(s))
        {
            atomicOr(overflow_flag, 1);
            s = 0.0;
        }
    }
    sh[tid] = s;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (tid < stride) sh[tid] = fmax(sh[tid], sh[tid + stride]);
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
    TB s = static_cast<TB>(0.0);
    for (int k = rp[i]; k < rp[i + 1]; ++k) s += static_cast<TB>(aval[k]) * __ldg(&x[ci[k]]);
    r[i] = b[i] - s;
}

// Pass 1 (Zero-Duplicate-Index):
//   !WIDE: 4 B/row scale-factored y_i = s_i * t_i with runtime overflow/subnormal oor flag.
//    WIDE: 8 B/row unscaled double t_i (scaled by pass1_scale when 0 < omega <= 1 to avoid Pass-2 overflow).
template <Mode M, bool WIDE, typename TB>
__global__ void bitfsai_fwd_kernel(
    int n, double pass1_scale,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const TB* __restrict__ r,
    float* __restrict__ y,
    double* __restrict__ t_f64,
    int* __restrict__ oor_flag)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int rs = rp[i], re = rp[i + 1];

    if constexpr (M == Mode::Fp64)
    {
        double acc = g_diag_f64[i] * static_cast<double>(r[i]);
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i) acc += g_f64[k] * static_cast<double>(__ldg(&r[col]));
        }
        t_f64[i] = pass1_scale * acc;
    }
    else if constexpr (WIDE)
    {
        // Standard IEEE-754 binary64 normal-range Gram accumulation.
        double sc = static_cast<double>(g_off_scale[i]);
        double acc = static_cast<double>(g_diag[i]) * static_cast<double>(r[i]);
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                double w = static_cast<double>(off_weight<M>(k, q_packed, g_f16));
                acc = fma(sc * w, static_cast<double>(__ldg(&r[col])), acc);
            }
        }
        t_f64[i] = pass1_scale * acc;
    }
    else
    {
        float sc = g_off_scale[i];
        float r_i = static_cast<float>(r[i]);
        bool any_nz = (r_i != 0.0f);
        float acc = 0.0f;
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col < i)
            {
                float w = off_weight<M>(k, q_packed, g_f16);
                float r_c = static_cast<float>(__ldg(&r[col]));
                if (w != 0.0f && r_c != 0.0f) any_nz = true;
                acc = fmaf(w, r_c, acc);
            }
        }
        float t_i = fmaf(sc, acc, g_diag[i] * r_i);
        float y_i = sc * t_i;
        y[i] = y_i;
        if (!isfinite(y_i) ||
            (t_i != 0.0f && (fabsf(t_i) < FLT_MIN || fabsf(y_i) < FLT_MIN)) ||
            (any_nz && fabsf(t_i) < FLT_MIN))
        {
            atomicOr(oor_flag, 1);
        }
    }
}

// Pass 2 (Zero-Duplicate-Index): x[j] = (X_IS_ZERO ? 0 : x[j]) + omega * (G_q^T * t)_j
template <Mode M, bool WIDE, bool X_IS_ZERO, typename TB>
__global__ void bitfsai_tr_kernel(
    int n, double omega,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const double* __restrict__ g_diag_f64,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16, const double* __restrict__ g_f64,
    const float* __restrict__ y,
    const double* __restrict__ t_f64,
    TB* __restrict__ x,
    const int* __restrict__ oor_flag)
{
    if constexpr (!WIDE && M != Mode::Fp64)
    {
        if (*oor_flag != 0) return;
    }
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int rs = rp[j], re = rp[j + 1];

    double sum_d;
    if constexpr (M == Mode::Fp64)
    {
        double acc = g_diag_f64[j] * t_f64[j];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j) acc += g_f64[k] * __ldg(&t_f64[col]);
        }
        sum_d = acc;
    }
    else if constexpr (WIDE)
    {
        double acc = static_cast<double>(g_diag[j]) * t_f64[j];
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j)
            {
                double w = static_cast<double>(off_weight<M>(k, q_packed, g_f16));
                acc = fma(static_cast<double>(__ldg(&g_off_scale[col])) * w, __ldg(&t_f64[col]), acc);
            }
        }
        sum_d = acc;
    }
    else
    {
        float acc = g_diag[j] * (y[j] / g_off_scale[j]);
        for (int k = rs; k < re; ++k)
        {
            int col = ci[k];
            if (col > j) acc = fmaf(off_weight<M>(k, q_packed, g_f16), __ldg(&y[col]), acc);
        }
        if (!isfinite(acc) || (acc != 0.0f && fabsf(acc) < FLT_MIN))
        {
            double dsum = static_cast<double>(g_diag[j]) *
                          (static_cast<double>(y[j]) / static_cast<double>(g_off_scale[j]));
            for (int k = rs; k < re; ++k)
            {
                int col = ci[k];
                if (col > j)
                {
                    dsum = fma(static_cast<double>(off_weight<M>(k, q_packed, g_f16)),
                               static_cast<double>(__ldg(&y[col])), dsum);
                }
            }
            sum_d = dsum;
        }
        else
        {
            sum_d = static_cast<double>(acc);
        }
    }
    double update = omega * sum_d;
    if constexpr (X_IS_ZERO) x[j] = static_cast<TB>(update);
    else x[j] = static_cast<TB>(static_cast<double>(x[j]) + update);
}

// Predicated double-precision fallback kernels when ordinary FP32 Pass 1 flags oor.
template <Mode M, typename TB>
__global__ void bitfsai_fallback_fwd_kernel(
    int n, double pass1_scale,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16,
    const TB* __restrict__ r, double* __restrict__ dtmp,
    const int* __restrict__ oor_flag)
{
    if (*oor_flag == 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sc = static_cast<double>(g_off_scale[i]);
    double acc = static_cast<double>(g_diag[i]) * static_cast<double>(r[i]);
    for (int k = rp[i], re = rp[i + 1]; k < re; ++k)
    {
        int col = ci[k];
        if (col < i)
        {
            double w = static_cast<double>(off_weight<M>(k, q_packed, g_f16));
            acc = fma(sc * w, static_cast<double>(__ldg(&r[col])), acc);
        }
    }
    dtmp[i] = pass1_scale * acc;
}

template <Mode M, bool X_IS_ZERO, typename TB>
__global__ void bitfsai_fallback_tr_kernel(
    int n, double omega,
    const int* __restrict__ rp, const int* __restrict__ ci,
    const float* __restrict__ g_diag, const float* __restrict__ g_off_scale,
    const uint8_t* __restrict__ q_packed, const __half* __restrict__ g_f16,
    const double* __restrict__ dtmp, TB* __restrict__ x,
    const int* __restrict__ oor_flag)
{
    if (*oor_flag == 0) return;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double acc = static_cast<double>(g_diag[j]) * dtmp[j];
    for (int k = rp[j], re = rp[j + 1]; k < re; ++k)
    {
        int col = ci[k];
        if (col > j)
        {
            double w = static_cast<double>(off_weight<M>(k, q_packed, g_f16));
            acc = fma(static_cast<double>(__ldg(&g_off_scale[col])) * w, __ldg(&dtmp[col]), acc);
        }
    }
    double update = omega * acc;
    if constexpr (X_IS_ZERO) x[j] = static_cast<TB>(update);
    else x[j] = static_cast<TB>(static_cast<double>(x[j]) + update);
}

struct DeviceScratch
{
    uint8_t* q_tmp = nullptr;
    int*     flags = nullptr;
    double*  u = nullptr;
    double*  v = nullptr;
    double*  max_ub = nullptr;
    bool     async_q_tmp = false;
    bool     async_flags = false;

    static cudaError_t alloc_fast(void** ptr, size_t bytes, bool& is_async)
    {
        is_async = false;
#if CUDART_VERSION >= 11020
        cudaError_t err = cudaMallocAsync(ptr, bytes, 0);
        if (err == cudaSuccess)
        {
            is_async = true;
            return cudaSuccess;
        }
        if (err == cudaErrorNotSupported || err == cudaErrorInsufficientDriver)
        {
            (void)cudaGetLastError();
        }
        else
        {
            return err;
        }
#endif
        return cudaMalloc(ptr, bytes);
    }

    static void free_fast(void* ptr, bool is_async)
    {
        if (!ptr) return;
#if CUDART_VERSION >= 11020
        if (is_async)
        {
            (void)cudaFreeAsync(ptr, 0);
            return;
        }
#endif
        (void)cudaFree(ptr);
    }

    ~DeviceScratch()
    {
        free_fast(q_tmp, async_q_tmp);
        free_fast(flags, async_flags);
        if (u)      cudaFree(u);
        if (v)      cudaFree(v);
        if (max_ub) cudaFree(max_ub);
    }
};

} // namespace bitfsai_detail

template <class T_Config>
BitFSAISolver_Base<T_Config>::BitFSAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope)
{
    std::string mode_str = cfg.AMG_Config::template getParameter<std::string>("bitfsai_mode", cfg_scope);
    if      (mode_str == "int4_split") m_mode = Mode::Int4Split;
    else if (mode_str == "fp4_e2m1")   m_mode = Mode::Fp4E2m1;
    else if (mode_str == "fp16")       m_mode = Mode::Fp16;
    else if (mode_str == "fp64")       m_mode = Mode::Fp64;
    else FatalError("Unknown bitfsai_mode value (expected int4_split, fp4_e2m1, fp16, or fp64)",
                    AMGX_ERR_CONFIGURATION);

    m_omega_cfg = cfg.AMG_Config::template getParameter<double>("bitfsai_omega", cfg_scope);
    if (!std::isfinite(m_omega_cfg) || !(m_omega_cfg > 0.0))
    {
        FatalError("bitfsai_omega must be finite and strictly positive", AMGX_ERR_CONFIGURATION);
    }
    m_omega = m_omega_cfg;

    m_max_patch_cfg = cfg.AMG_Config::template getParameter<int>("bitfsai_max_patch", cfg_scope);
    if (m_max_patch_cfg < 0 || m_max_patch_cfg > bitfsai_detail::MAX_PATCH)
    {
        FatalError("bitfsai_max_patch must be in [0, 16]", AMGX_ERR_CONFIGURATION);
    }
    m_use_safe_omega = (cfg.AMG_Config::template getParameter<int>("bitfsai_safe_omega", cfg_scope) != 0);
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
    if (d_work)        cudaFree(d_work);
    if (d_res)         cudaFree(d_res);
    d_g_diag = nullptr;
    d_g_off_scale = nullptr;
    d_q_packed = nullptr;
    d_g_f16 = nullptr;
    d_g_diag_f64 = nullptr;
    d_g_f64 = nullptr;
    d_work = nullptr;
    d_res = nullptr;
    m_n = 0;
    m_nnz = 0;
    m_alloc_mode = -1;
    m_alloc_wide = false;
    m_wide_apply = false;
}

template <class T_Config>
BitFSAISolver_Base<T_Config>::~BitFSAISolver_Base()
{
    free_buffers();
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

        const int n = static_cast<int>(A.get_num_rows());
        const int nnz = static_cast<int>(A.get_num_nz());
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

        constexpr uint64_t kMaxGridX = 2147483647ULL;
        const uint64_t n_u64 = static_cast<uint64_t>(n);
        const uint64_t nnz_u64 = static_cast<uint64_t>(nnz);
        const uint64_t th_u64 = 256ULL;
        const uint64_t bl_u64 = (n_u64 + th_u64 - 1ULL) / th_u64;
        const uint64_t packed_u64 = (nnz_u64 + 1ULL) / 2ULL;
        const uint64_t pack_bl_u64 = (packed_u64 + th_u64 - 1ULL) / th_u64;
        if (bl_u64 > kMaxGridX || pack_bl_u64 > kMaxGridX ||
            (n_u64 + 1ULL) > SIZE_MAX / sizeof(double) ||
            nnz_u64 > SIZE_MAX / sizeof(double))
        {
            this->free_buffers();
            FatalError("BitFSAISolver matrix dimensions exceed allocation or CUDA grid limits",
                       AMGX_ERR_BAD_PARAMETERS);
        }

        auto check_cuda = [&](cudaError_t err, const char* msg)
        {
            if (err != cudaSuccess)
            {
                this->free_buffers();
                FatalError(msg, AMGX_ERR_CUDA_FAILURE);
            }
        };

        const auto mode = static_cast<bitfsai_detail::Mode>(static_cast<int>(this->m_mode));
        const int mode_int = static_cast<int>(mode);
        const bool uses_4bit = (mode == bitfsai_detail::Mode::Int4Split || mode == bitfsai_detail::Mode::Fp4E2m1);
        const int th = 256;
        const int bl = static_cast<int>(bl_u64);
        const int avg_deg = nnz / std::max(n, 1);
        const int max_patch = (this->m_max_patch_cfg > 0) ? this->m_max_patch_cfg : ((avg_deg > 22) ? 8 : 12);

        const bool alloc_4bit = (this->m_alloc_mode == static_cast<int>(bitfsai_detail::Mode::Int4Split) ||
                                 this->m_alloc_mode == static_cast<int>(bitfsai_detail::Mode::Fp4E2m1));
        const bool layout_compat = (mode_int == this->m_alloc_mode) || (uses_4bit && alloc_4bit);
        const bool reuse_pattern = reuse && (n == this->m_n) && (nnz == this->m_nnz) && layout_compat;

        bitfsai_detail::DeviceScratch scratch;
        check_cuda(bitfsai_detail::DeviceScratch::alloc_fast(
                       reinterpret_cast<void**>(&scratch.flags), 2 * sizeof(int), scratch.async_flags),
                   "BitFSAISolver failed to allocate validation flags");
        check_cuda(cudaMemsetAsync(scratch.flags, 0, 2 * sizeof(int), 0),
                   "BitFSAISolver failed to zero validation flags");

        // Validate CSR offsets, column bounds, and duplicate-free rows on initial setup or pattern change
        if (!reuse_pattern)
        {
            bitfsai_detail::validate_csr_rows_kernel<<<bl, th>>>(n, nnz, A.row_offsets.raw(), scratch.flags);
            check_cuda(cudaGetLastError(), "BitFSAISolver CSR row validation launch failed");
            bitfsai_detail::validate_csr_cols_kernel<<<bl, th>>>(
                n, A.row_offsets.raw(), A.col_indices.raw(), scratch.flags, scratch.flags + 1);
            check_cuda(cudaGetLastError(), "BitFSAISolver CSR column validation launch failed");

            int h_csr_err[2] = {0, 0};
            check_cuda(cudaMemcpy(h_csr_err, scratch.flags, 2 * sizeof(int), cudaMemcpyDeviceToHost),
                       "BitFSAISolver CSR validation copy failed");
            if ((h_csr_err[0] | h_csr_err[1]) & bitfsai_detail::kErrInvalidCsr)
            {
                this->free_buffers();
                FatalError("BitFSAISolver requires valid monotonic CSR row offsets, in-range columns, and no duplicate columns",
                           AMGX_ERR_BAD_PARAMETERS);
            }
        }

        if (!reuse_pattern)
        {
            this->free_buffers();
            check_cuda(cudaMalloc(&this->d_res, static_cast<size_t>(n_u64) * sizeof(ValueTypeB)),
                       "BitFSAISolver failed to allocate residual buffer");
            if (mode == bitfsai_detail::Mode::Fp64)
            {
                check_cuda(cudaMalloc(&this->d_g_diag_f64, static_cast<size_t>(n_u64) * sizeof(double)),
                           "BitFSAISolver failed to allocate FP64 diagonal buffer");
                check_cuda(cudaMalloc(&this->d_g_f64, static_cast<size_t>(nnz_u64) * sizeof(double)),
                           "BitFSAISolver failed to allocate FP64 factor buffer");
            }
            else
            {
                check_cuda(cudaMalloc(&this->d_g_diag, static_cast<size_t>(n_u64) * sizeof(float)),
                           "BitFSAISolver failed to allocate FP32 diagonal buffer");
                check_cuda(cudaMalloc(&this->d_g_off_scale, static_cast<size_t>(n_u64) * sizeof(float)),
                           "BitFSAISolver failed to allocate FP32 scale buffer");
                if (mode == bitfsai_detail::Mode::Fp16)
                {
                    check_cuda(cudaMalloc(&this->d_g_f16, static_cast<size_t>(nnz_u64) * sizeof(__half)),
                               "BitFSAISolver failed to allocate FP16 factor buffer");
                }
                else
                {
                    check_cuda(cudaMalloc(&this->d_q_packed, static_cast<size_t>(packed_u64)),
                               "BitFSAISolver failed to allocate 4-bit packed factor buffer");
                }
            }
        }
        else
        {
            // Invalidate reusable state until setup validation succeeds
            this->m_n = 0;
            this->m_nnz = 0;
            this->m_alloc_mode = -1;
        }

        const bool check_full_sym = !reuse_pattern;
        if (uses_4bit)
        {
            check_cuda(bitfsai_detail::DeviceScratch::alloc_fast(
                           reinterpret_cast<void**>(&scratch.q_tmp), static_cast<size_t>(nnz_u64),
                           scratch.async_q_tmp),
                       "BitFSAISolver failed to allocate 4-bit packing staging buffer");
            const uint8_t zero_code = (mode == bitfsai_detail::Mode::Fp4E2m1) ? 0u : 8u;
            check_cuda(cudaMemsetAsync(scratch.q_tmp, zero_code, static_cast<size_t>(nnz_u64), 0),
                       "BitFSAISolver failed to initialize 4-bit staging buffer");
            bitfsai_detail::bitfsai_build_kernel<<<bl, th>>>(
                n, mode, max_patch, this->m_use_reproject, check_full_sym,
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                scratch.q_tmp, this->d_g_f16, this->d_g_f64, scratch.flags);
            check_cuda(cudaGetLastError(), "BitFSAISolver build kernel launch failed");
            bitfsai_detail::pack_nibbles_kernel<<<static_cast<int>(pack_bl_u64), th>>>(
                nnz, scratch.q_tmp, this->d_q_packed);
            check_cuda(cudaGetLastError(), "BitFSAISolver pack kernel launch failed");
        }
        else
        {
            if (mode == bitfsai_detail::Mode::Fp16)
            {
                check_cuda(cudaMemsetAsync(this->d_g_f16, 0, static_cast<size_t>(nnz_u64) * sizeof(__half), 0),
                           "BitFSAISolver failed to zero FP16 factor buffer");
            }
            else
            {
                check_cuda(cudaMemsetAsync(this->d_g_f64, 0, static_cast<size_t>(nnz_u64) * sizeof(double), 0),
                           "BitFSAISolver failed to zero FP64 factor buffer");
            }
            bitfsai_detail::bitfsai_build_kernel<<<bl, th>>>(
                n, mode, max_patch, this->m_use_reproject, check_full_sym,
                A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                nullptr, this->d_g_f16, this->d_g_f64, scratch.flags);
            check_cuda(cudaGetLastError(), "BitFSAISolver build kernel launch failed");
        }

        int h_err = 0;
        check_cuda(cudaMemcpy(&h_err, scratch.flags, sizeof(int), cudaMemcpyDeviceToHost),
                   "BitFSAISolver build status copy failed");

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
        if (h_err & bitfsai_detail::kErrInvalidValue)
        {
            this->free_buffers();
            FatalError("BitFSAISolver requires finite matrix values", AMGX_ERR_BAD_PARAMETERS);
        }
        if (h_err & bitfsai_detail::kErrUnrepresentable)
        {
            this->free_buffers();
            FatalError("BitFSAISolver factor cannot be represented in FP32", AMGX_ERR_BAD_PARAMETERS);
        }

        const bool wide_apply = (mode == bitfsai_detail::Mode::Fp64) ||
                                std::is_same_v<ValueTypeB, double> ||
                                ((h_err & bitfsai_detail::kFlagWideApply) != 0);
        const size_t work_bytes = wide_apply
                                      ? static_cast<size_t>(n_u64) * sizeof(double)
                                      : static_cast<size_t>(n_u64 + 1ULL) * sizeof(double);
        if (!this->d_work || this->m_alloc_wide != wide_apply || !reuse)
        {
            if (this->d_work)
            {
                cudaFree(this->d_work);
                this->d_work = nullptr;
            }
            check_cuda(cudaMalloc(&this->d_work, work_bytes),
                       "BitFSAISolver failed to allocate apply workspace");
            this->m_alloc_wide = wide_apply;
        }
        this->m_wide_apply = wide_apply;

        this->m_omega = this->m_omega_cfg;
        if (this->m_use_safe_omega)
        {
            check_cuda(cudaMalloc(&scratch.u, static_cast<size_t>(n_u64) * sizeof(double)),
                       "BitFSAISolver safe-omega allocation failed (u)");
            check_cuda(cudaMalloc(&scratch.v, static_cast<size_t>(n_u64) * sizeof(double)),
                       "BitFSAISolver safe-omega allocation failed (v)");
            check_cuda(cudaMalloc(&scratch.max_ub, sizeof(double)),
                       "BitFSAISolver safe-omega allocation failed (max)");
            check_cuda(cudaMemset(scratch.max_ub, 0, sizeof(double)),
                       "BitFSAISolver safe-omega memset failed (max)");
            check_cuda(cudaMemset(scratch.flags, 0, sizeof(int)),
                       "BitFSAISolver safe-omega memset failed (flags)");

            auto run_gersh = [&](auto mode_tag)
            {
                constexpr bitfsai_detail::Mode M = decltype(mode_tag)::value;
                bitfsai_detail::gershgorin_abs_gt1_kernel<M><<<bl, th>>>(
                    n, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    scratch.u, scratch.flags);
                check_cuda(cudaGetLastError(), "BitFSAISolver safe-omega pass 1 failed");
                bitfsai_detail::gershgorin_abs_au_kernel<<<bl, th>>>(
                    n, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                    scratch.u, scratch.v, scratch.flags);
                check_cuda(cudaGetLastError(), "BitFSAISolver safe-omega pass 2 failed");
                bitfsai_detail::gershgorin_abs_gv_max_kernel<M><<<bl, th>>>(
                    n, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    scratch.v, scratch.max_ub, scratch.flags);
                check_cuda(cudaGetLastError(), "BitFSAISolver safe-omega pass 3 failed");
            };
            if      (mode == bitfsai_detail::Mode::Int4Split) run_gersh(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Int4Split>{});
            else if (mode == bitfsai_detail::Mode::Fp4E2m1)   run_gersh(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp4E2m1>{});
            else if (mode == bitfsai_detail::Mode::Fp16)      run_gersh(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp16>{});
            else                                              run_gersh(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp64>{});

            int h_ovf = 0;
            double h_ub = 0.0;
            check_cuda(cudaMemcpy(&h_ovf, scratch.flags, sizeof(int), cudaMemcpyDeviceToHost),
                       "BitFSAISolver safe-omega flag copy failed");
            check_cuda(cudaMemcpy(&h_ub, scratch.max_ub, sizeof(double), cudaMemcpyDeviceToHost),
                       "BitFSAISolver safe-omega bound copy failed");
            if (h_ovf != 0 || !std::isfinite(h_ub) || !(h_ub > 0.0))
            {
                this->free_buffers();
                FatalError("BitFSAISolver safe-omega Gershgorin bound overflowed or was non-positive",
                           AMGX_ERR_BAD_PARAMETERS);
            }
            // Conservative lower bound on 1 / U_upper so omega * lambda_max(M * A) <= 1.0 strictly.
            double omega_cap = std::nextafter(1.0 / h_ub, 0.0);
            while (omega_cap > 0.0 && h_ub * omega_cap > 1.0)
            {
                omega_cap = std::nextafter(omega_cap, 0.0);
            }
            if (!(omega_cap > 0.0))
            {
                this->free_buffers();
                FatalError("BitFSAISolver safe-omega bound underflowed to zero", AMGX_ERR_BAD_PARAMETERS);
            }
            this->m_omega = std::min(this->m_omega_cfg, omega_cap);
        }

        this->m_n = n;
        this->m_nnz = nnz;
        this->m_alloc_mode = mode_int;
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
        int n = static_cast<int>(A.get_num_rows());
        if (n <= 0) return;
        if (this->m_n != n || this->m_alloc_mode < 0 || !this->d_res || !this->d_work)
        {
            FatalError("BitFSAISolver is not initialized; setup must succeed before solve", AMGX_ERR_BAD_PARAMETERS);
        }

        auto check_cuda = [&](cudaError_t err, const char* msg)
        {
            if (err != cudaSuccess) FatalError(msg, AMGX_ERR_CUDA_FAILURE);
        };

        const int th = 256;
        const int bl = static_cast<int>((static_cast<uint64_t>(n) + static_cast<uint64_t>(th) - 1ULL) /
                                        static_cast<uint64_t>(th));
        const ValueTypeB* r_ptr = b.raw();
        if (!xIsZero)
        {
            bitfsai_detail::fused_residual_kernel<<<bl, th>>>(
                n, A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                b.raw(), x.raw(), this->d_res);
            check_cuda(cudaGetLastError(), "BitFSAISolver fused residual launch failed");
            r_ptr = this->d_res;
        }

        const double om = this->m_omega;
        const auto mode = static_cast<bitfsai_detail::Mode>(static_cast<int>(this->m_mode));
        const bool wide = this->m_wide_apply;
        const bool damp_pass1 = (om > 0.0 && om <= 1.0);
        const double p1_scale = damp_pass1 ? om : 1.0;
        const double p2_scale = damp_pass1 ? 1.0 : om;

        auto launch_mode = [&](auto mode_tag, auto xzero_tag)
        {
            constexpr bitfsai_detail::Mode M = decltype(mode_tag)::value;
            constexpr bool XZ = decltype(xzero_tag)::value;
            if (wide || M == bitfsai_detail::Mode::Fp64)
            {
                double* t_f64 = static_cast<double*>(this->d_work);
                bitfsai_detail::bitfsai_fwd_kernel<M, true><<<bl, th>>>(
                    n, p1_scale, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    r_ptr, nullptr, t_f64, nullptr);
                check_cuda(cudaGetLastError(), "BitFSAISolver wide forward launch failed");
                bitfsai_detail::bitfsai_tr_kernel<M, true, XZ><<<bl, th>>>(
                    n, p2_scale, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    nullptr, t_f64, x.raw(), nullptr);
                check_cuda(cudaGetLastError(), "BitFSAISolver wide transpose launch failed");
            }
            else
            {
                float*  y = static_cast<float*>(this->d_work);
                double* dtmp = static_cast<double*>(this->d_work);
                int*    oor_flag = reinterpret_cast<int*>(dtmp + n);
                check_cuda(cudaMemsetAsync(oor_flag, 0, sizeof(int), 0),
                           "BitFSAISolver oor_flag memset failed");
                bitfsai_detail::bitfsai_fwd_kernel<M, false><<<bl, th>>>(
                    n, 1.0, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    r_ptr, y, nullptr, oor_flag);
                check_cuda(cudaGetLastError(), "BitFSAISolver forward launch failed");
                bitfsai_detail::bitfsai_tr_kernel<M, false, XZ><<<bl, th>>>(
                    n, om, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale, this->d_g_diag_f64,
                    this->d_q_packed, this->d_g_f16, this->d_g_f64,
                    y, nullptr, x.raw(), oor_flag);
                check_cuda(cudaGetLastError(), "BitFSAISolver transpose launch failed");
                bitfsai_detail::bitfsai_fallback_fwd_kernel<M><<<bl, th>>>(
                    n, p1_scale, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale,
                    this->d_q_packed, this->d_g_f16,
                    r_ptr, dtmp, oor_flag);
                check_cuda(cudaGetLastError(), "BitFSAISolver fallback forward launch failed");
                bitfsai_detail::bitfsai_fallback_tr_kernel<M, XZ><<<bl, th>>>(
                    n, p2_scale, A.row_offsets.raw(), A.col_indices.raw(),
                    this->d_g_diag, this->d_g_off_scale,
                    this->d_q_packed, this->d_g_f16,
                    dtmp, x.raw(), oor_flag);
                check_cuda(cudaGetLastError(), "BitFSAISolver fallback transpose launch failed");
            }
        };

        auto dispatch = [&](auto mode_tag)
        {
            if (xIsZero) launch_mode(mode_tag, std::integral_constant<bool, true>{});
            else         launch_mode(mode_tag, std::integral_constant<bool, false>{});
        };

        if      (mode == bitfsai_detail::Mode::Int4Split) dispatch(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Int4Split>{});
        else if (mode == bitfsai_detail::Mode::Fp4E2m1)   dispatch(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp4E2m1>{});
        else if (mode == bitfsai_detail::Mode::Fp16)      dispatch(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp16>{});
        else                                              dispatch(std::integral_constant<bitfsai_detail::Mode, bitfsai_detail::Mode::Fp64>{});
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
