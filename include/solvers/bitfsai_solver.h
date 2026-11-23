// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <solvers/solver.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace amgx
{

template <class T_Config> class BitFSAISolver;

// Zero-Duplicate-Index Sub-Byte Factorized Sparse Approximate Inverse (BitFSAI) Smoother.
//
// Factor encodings (configured via `bitfsai_mode`):
//   int4_split (default): Split-scale symmetric signed INT4 off-diagonals + FP32 diagonal (0.5 B/nnz)
//   nvfp4               : Split-scale FP4 (E2M1) off-diagonals + FP32 diagonal (0.5 B/nnz)
//   int4                : Diagonal-inclusive uniform signed INT4 (0.5 B/nnz; ablation mode)
//   fp16                : Split-scale FP16 off-diagonals + FP32 diagonal (2.0 B/nnz)
//   fp64                : Uncompressed FP64 reference (8.0 B/nnz)
//   auto                : Per-level local patch A-energy selector between int4_split and nvfp4
//
// Supported AMGX config parameters (registered in src/core.cu):
//   bitfsai_mode         = "int4_split" | "nvfp4" | "int4" | "fp16" | "fp64" | "auto"
//   bitfsai_omega        = damping factor (default 0.72)
//   bitfsai_max_patch    = max local Cholesky patch size 1..16 (0 = auto-adaptive 8/12)
//   bitfsai_safe_omega   = 1 to cap damping via certified Gershgorin upper bound || |G_q||A||G_q|^T 1 ||_inf
//   bitfsai_free_scratch = 1 to release per-device staging workspace immediately after packing
//   bitfsai_reproject    = 1 to apply post-quantization 2x2 A-energy optimal re-projection (default 1)
template <class T_Config>
class BitFSAISolver_Base : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef Vector<T_Config> VVector;

    protected:
        Matrix<T_Config> *m_explicit_A = nullptr;
        int    m_mode = 4;               // Default: int4_split
        int    m_active_mode = 4;        // Resolved mode for this level (used by auto)
        int    m_max_patch_cfg = 0;      // 0 = auto-adaptive (8 for avg_deg > 22, else 12)
        double m_omega_cfg = 0.72;       // Configured base damping
        double m_omega = 0.72;           // Effective damping (may be reduced by safe_omega)
        bool   m_use_safe_omega = false;
        bool   m_free_scratch = false;
        bool   m_use_reproject = true;
        bool   m_wide_apply = false;

        int m_n = 0;
        int m_nnz = 0;
        int m_alloc_mode = -1;
        int m_diag_fallback_rows = 0;    // Count of rows falling back to diagonal Jacobi in last setup

        void free_buffers();

        // Zero-duplicate-index persistent GPU factor storage (aligned 1-to-1 with A's CSR):
        float*    d_g_diag = nullptr;      // FP32 diagonal g_ii (modes 0,1,3,4,5)
        float*    d_g_off_scale = nullptr; // FP32 per-row scale s_i (modes 0,1,3,4,5)
        uint8_t*  d_q_packed = nullptr;    // 4-bit packed nibbles (ceil(nnz/2) bytes, modes 0,3,4,5)
        __half*   d_g_f16 = nullptr;       // FP16 weights (2*nnz bytes, mode 1)

        // True FP64 reference buffers (mode 2 only):
        double*   d_g_diag_f64 = nullptr;  // FP64 diagonal g_ii
        double*   d_g_f64 = nullptr;       // FP64 off-diagonal weights (8*nnz bytes)
        double*   d_t_f64 = nullptr;       // FP64 Pass-1 intermediate

        // Apply workspace buffers:
        float*      d_y = nullptr;         // Scale-factored Pass-1 intermediate y_i = s_i * t_i (4 B/row)
        ValueTypeB* d_res = nullptr;       // Fused residual buffer (sized to ValueTypeB)

        virtual void setup_device(const Matrix<T_Config> &A, bool reuse) = 0;
        virtual void smooth_device(const Matrix<T_Config> &A, const VVector &b, VVector &x, bool xIsZero) = 0;

    public:
        BitFSAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope);
        ~BitFSAISolver_Base();

        bool is_residual_needed() const { return false; }
        void solver_setup(bool reuse_matrix_structure);
        bool isColoringNeeded() const { return false; }
        bool getReorderColsByColorDesired() const { return false; }
        bool getInsertDiagonalDesired() const { return false; }
        void solve_init(VVector &b, VVector &x, bool xIsZero) {}
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x) {}
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BitFSAISolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
    : public BitFSAISolver_Base<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef Vector<TConfig_h> VVector;
        BitFSAISolver(AMG_Config &cfg, const std::string &cfg_scope)
            : BitFSAISolver_Base<TConfig_h>(cfg, cfg_scope) {}
    protected:
        void setup_device(const Matrix_h &A, bool reuse) override
        {
            FatalError("BitFSAISolver does not support host execution mode", AMGX_ERR_NOT_IMPLEMENTED);
        }
        void smooth_device(const Matrix_h &A, const VVector &b, VVector &x, bool xIsZero) override
        {
            FatalError("BitFSAISolver does not support host execution mode", AMGX_ERR_NOT_IMPLEMENTED);
        }
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class BitFSAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
    : public BitFSAISolver_Base<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef Vector<TConfig_d> VVector;
        BitFSAISolver(AMG_Config &cfg, const std::string &cfg_scope)
            : BitFSAISolver_Base<TConfig_d>(cfg, cfg_scope) {}
    protected:
        void setup_device(const Matrix_d &A, bool reuse) override;
        void smooth_device(const Matrix_d &A, const VVector &b, VVector &x, bool xIsZero) override;
};

template <class T_Config>
class BitFSAISolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        {
            return new BitFSAISolver<T_Config>(cfg, cfg_scope);
        }
};

void release_bitfsai_staging_pool();

} // namespace amgx
