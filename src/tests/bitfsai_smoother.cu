// SPDX-FileCopyrightText: 2026 Brett Bonner
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "test_utils.h"
#include "amg_solver.h"
#include "solvers/solver.h"
#include "solvers/bitfsai_solver.h"
#include <multiply.h>
#include <blas.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(BitFSAISmoother);

static void build_anisotropic_3d_csr(
    int N, double az, double scale,
    std::vector<int> &rp, std::vector<int> &ci, std::vector<double> &val)
{
    int n = N * N * N;
    rp.assign(n + 1, 0);
    ci.clear();
    val.clear();
    auto idx = [&](int x, int y, int z) { return (z * N + y) * N + x; };
    for (int z = 0; z < N; ++z)
    {
        for (int y = 0; y < N; ++y)
        {
            for (int x = 0; x < N; ++x)
            {
                int r = idx(x, y, z);
                double diag = 1e-3;
                std::vector<std::pair<int, double>> row;
                auto add = [&](int nx, int ny, int nz, double w)
                {
                    diag += w;
                    if (nx >= 0 && nx < N && ny >= 0 && ny < N && nz >= 0 && nz < N)
                        row.push_back({idx(nx, ny, nz), -w * scale});
                };
                add(x - 1, y, z, 1.0);
                add(x + 1, y, z, 1.0);
                add(x, y - 1, z, 1.0);
                add(x, y + 1, z, 1.0);
                add(x, y, z - 1, az);
                add(x, y, z + 1, az);
                row.push_back({r, diag * scale});
                std::sort(row.begin(), row.end());
                for (const auto &kv : row)
                {
                    ci.push_back(kv.first);
                    val.push_back(kv.second);
                }
                rp[r + 1] = static_cast<int>(ci.size());
            }
        }
    }
}

void upload_csr_to_device(
    int n, const std::vector<int> &rp, const std::vector<int> &ci, const std::vector<double> &val,
    Matrix<TConfig> &A_d)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    Matrix<TConfig_h> A_h;
    A_h.set_initialized(0);
    A_h.addProps(CSR);
    A_h.resize(n, n, static_cast<int>(ci.size()), 1, 1, 0);
    for (int i = 0; i <= n; ++i) A_h.row_offsets[i] = rp[i];
    for (size_t k = 0; k < ci.size(); ++k)
    {
        A_h.col_indices[k] = ci[k];
        A_h.values[k] = static_cast<typename TConfig_h::MatPrec>(val[k]);
    }
    A_h.set_initialized(1);
    A_d = A_h;
}

void run_2x2_case(
    const std::string &mode, double a00, double a01, double a11,
    double b0, double b1, double exp0, double exp1, double rel_tol,
    double omega = 1.0, int safe_omega = 0)
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;
    std::vector<int> rp = {0, 2, 4};
    std::vector<int> ci = {0, 1, 0, 1};
    std::vector<double> val = {a00, a01, a01, a11};
    Matrix<TConfig> A_d;
    upload_csr_to_device(2, rp, ci, val, A_d);

    Vector<TConfig_h> b_h(2), x_h(2);
    b_h[0] = static_cast<ValueTypeB>(b0);
    b_h[1] = static_cast<ValueTypeB>(b1);
    x_h[0] = static_cast<ValueTypeB>(0.0);
    x_h[1] = static_cast<ValueTypeB>(0.0);
    Vector<TConfig> b_d = b_h;
    Vector<TConfig> x_d = x_h;

    AMG_Config cfg;
    std::string pstr = "solver=BITFSAI, bitfsai_mode=" + mode +
                       ", bitfsai_omega=" + std::to_string(omega) +
                       ", bitfsai_safe_omega=" + std::to_string(safe_omega) +
                       ", max_iters=1";
    UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
    Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    smoother->setup(A_d, false);
    smoother->solve(b_d, x_d, true);
    x_h = x_d;
    delete smoother;

    double z0 = static_cast<double>(x_h[0]);
    double z1 = static_cast<double>(x_h[1]);
    double rz = b0 * z0 + b1 * z1;
    UNITTEST_ASSERT_TRUE(std::isfinite(z0) && std::isfinite(z1));
    UNITTEST_ASSERT_TRUE(rz > 0.0);
    double err0 = std::fabs(z0 - exp0) / std::max(1e-300, std::fabs(exp0));
    double err1 = std::fabs(z1 - exp1) / std::max(1e-300, std::fabs(exp1));
    UNITTEST_ASSERT_TRUE(err0 <= rel_tol);
    UNITTEST_ASSERT_TRUE(err1 <= rel_tol);
}

void expect_setup_failure(Matrix<TConfig> &A_d, const std::string &mode)
{
    AMG_Config cfg;
    std::string pstr = "solver=BITFSAI, bitfsai_mode=" + mode + ", bitfsai_omega=1.0, max_iters=1";
    UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
    Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
    UNITTEST_ASSERT_TRUE(smoother != NULL);
    bool threw = false;
    try
    {
        smoother->setup(A_d, false);
    }
    catch (const amgx_exception &)
    {
        threw = true;
    }
    delete smoother;
    UNITTEST_ASSERT_TRUE(threw);
}

void run()
{
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;

    // 1. Test all 4 encodings with zero/nonzero initial guesses, safe_omega, and reuse=true resetup
    std::vector<int> rp, ci;
    std::vector<double> val, val_scaled;
    build_anisotropic_3d_csr(6, 50.0, 1.0, rp, ci, val);
    build_anisotropic_3d_csr(6, 5.0, 0.25, rp, ci, val_scaled);
    int n = static_cast<int>(rp.size()) - 1;

    Matrix<TConfig> A_d, A_scaled_d;
    upload_csr_to_device(n, rp, ci, val, A_d);
    upload_csr_to_device(n, rp, ci, val_scaled, A_scaled_d);

    Vector<TConfig_h> b_h(n), x0_h(n);
    for (int i = 0; i < n; ++i)
    {
        b_h[i] = static_cast<ValueTypeB>(std::sin(0.17 * (i + 1)));
        x0_h[i] = static_cast<ValueTypeB>(0.25 * std::cos(0.11 * (i + 1)));
    }
    Vector<TConfig> b_d = b_h;
    Vector<TConfig> r_d(n);
    Vector<TConfig_h> nrm_ini(1), nrm_fin(1);

    const char *modes[] = {"int4_split", "fp4_e2m1", "fp16", "fp64"};
    for (const char *mode : modes)
    {
        AMG_Config cfg;
        std::string pstr = std::string("solver=BITFSAI, bitfsai_mode=") + mode +
                           ", bitfsai_omega=0.72, bitfsai_safe_omega=1, max_iters=4";
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
        Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
        UNITTEST_ASSERT_TRUE(smoother != NULL);
        smoother->setup(A_d, false);

        // Zero initial guess
        Vector<TConfig> x_zero_d(n, static_cast<ValueTypeB>(0.0));
        smoother->solve(b_d, x_zero_d, true);
        multiply(A_d, x_zero_d, r_d);
        axpby(b_d, r_d, r_d, ValueTypeB(1), ValueTypeB(-1));
        get_norm(A_d, b_d, 1, L2, nrm_ini);
        get_norm(A_d, r_d, 1, L2, nrm_fin);
        UNITTEST_ASSERT_TRUE(std::isfinite(static_cast<double>(nrm_fin[0])));
        UNITTEST_ASSERT_TRUE(nrm_fin[0] < nrm_ini[0]);

        // Nonzero initial guess
        Vector<TConfig> x_nz_d = x0_h;
        multiply(A_d, x_nz_d, r_d);
        axpby(b_d, r_d, r_d, ValueTypeB(1), ValueTypeB(-1));
        get_norm(A_d, r_d, 1, L2, nrm_ini);
        smoother->solve(b_d, x_nz_d, false);
        multiply(A_d, x_nz_d, r_d);
        axpby(b_d, r_d, r_d, ValueTypeB(1), ValueTypeB(-1));
        get_norm(A_d, r_d, 1, L2, nrm_fin);
        UNITTEST_ASSERT_TRUE(std::isfinite(static_cast<double>(nrm_fin[0])));
        UNITTEST_ASSERT_TRUE(nrm_fin[0] < nrm_ini[0]);

        // Same-structure coefficient replacement (reuse = true)
        Matrix<TConfig> A_reuse_d = A_d;
        smoother->setup(A_reuse_d, false);
        A_reuse_d.values = A_scaled_d.values;
        smoother->setup(A_reuse_d, true);
        Vector<TConfig> x_reuse_d(n, static_cast<ValueTypeB>(0.0));
        smoother->solve(b_d, x_reuse_d, true);
        multiply(A_reuse_d, x_reuse_d, r_d);
        axpby(b_d, r_d, r_d, ValueTypeB(1), ValueTypeB(-1));
        get_norm(A_reuse_d, b_d, 1, L2, nrm_ini);
        get_norm(A_reuse_d, r_d, 1, L2, nrm_fin);
        UNITTEST_ASSERT_TRUE(std::isfinite(static_cast<double>(nrm_fin[0])));
        UNITTEST_ASSERT_TRUE(nrm_fin[0] < nrm_ini[0]);

        delete smoother;
    }

    // 2. End-to-end PCG + Classical AMG V-cycle with BITFSAI smoother
    {
        Resources res;
        AMG_Configuration cfg;
        std::string pstr =
            "config_version=2, solver(main)=PCG, main:preconditioner(amg)=AMG, "
            "amg:algorithm=CLASSICAL, amg:selector=PMIS, amg:interpolator=D2, amg:cycle=V, "
            "amg:smoother(sm)=BITFSAI, sm:bitfsai_mode=int4_split, sm:bitfsai_omega=0.72, "
            "amg:presweeps=1, amg:postsweeps=1, amg:max_iters=1, "
            "main:max_iters=200, main:tolerance=1e-6, main:norm=L2, main:convergence=RELATIVE_INI_CORE";
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(pstr.c_str()) == AMGX_OK);
        AMG_Solver<TConfig> amg_solver(&res, cfg);
        amg_solver.setup(A_d);
        Vector<TConfig> x_pcg_d(n, static_cast<ValueTypeB>(0.0));
        AMGX_STATUS st = AMGX_ST_CONVERGED;
        amg_solver.solve(b_d, x_pcg_d, st);
        UNITTEST_ASSERT_TRUE(st == AMGX_ST_CONVERGED);
        multiply(A_d, x_pcg_d, r_d);
        axpby(b_d, r_d, r_d, ValueTypeB(1), ValueTypeB(-1));
        get_norm(A_d, b_d, 1, L2, nrm_ini);
        get_norm(A_d, r_d, 1, L2, nrm_fin);
        UNITTEST_ASSERT_TRUE(static_cast<double>(nrm_fin[0]) / static_cast<double>(nrm_ini[0]) <= 1e-5);
    }

    // 3. Reject malformed / unsupported inputs cleanly:
    //    separate DIAG, zero-nnz, upper-only nonsymmetric, dropped-lower nonsymmetric, duplicate col, NaN
    {
        Matrix<TConfig_h> A_diag_h;
        A_diag_h.set_initialized(0);
        A_diag_h.addProps(CSR);
        A_diag_h.addProps(DIAG);
        A_diag_h.resize(2, 2, 0, 1, 1, 1);
        A_diag_h.row_offsets[0] = 0; A_diag_h.row_offsets[1] = 0; A_diag_h.row_offsets[2] = 0;
        A_diag_h.values[0] = static_cast<typename TConfig_h::MatPrec>(2.0);
        A_diag_h.values[1] = static_cast<typename TConfig_h::MatPrec>(2.0);
        A_diag_h.set_initialized(1);
        Matrix<TConfig> A_diag_d = A_diag_h;
        expect_setup_failure(A_diag_d, "int4_split");
    }
    {
        Matrix<TConfig_h> A_empty_h;
        A_empty_h.set_initialized(0);
        A_empty_h.addProps(CSR);
        A_empty_h.resize(2, 2, 0, 1, 1, 0);
        A_empty_h.row_offsets[0] = 0; A_empty_h.row_offsets[1] = 0; A_empty_h.row_offsets[2] = 0;
        A_empty_h.set_initialized(1);
        Matrix<TConfig> A_empty_d = A_empty_h;
        expect_setup_failure(A_empty_d, "int4_split");
    }
    {
        // Upper-only nonsymmetric pattern: (0,1) exists without (1,0)
        Matrix<TConfig> A_up_d;
        upload_csr_to_device(2, {0, 2, 3}, {0, 1, 1}, {2.0, -0.5, 2.0}, A_up_d);
        expect_setup_failure(A_up_d, "int4_split");
    }
    {
        // Lower-only nonsymmetric pattern: (1,0) exists without (0,1)
        Matrix<TConfig> A_lo_d;
        upload_csr_to_device(2, {0, 1, 3}, {0, 0, 1}, {2.0, -0.5, 2.0}, A_lo_d);
        expect_setup_failure(A_lo_d, "int4_split");
    }
    {
        // Duplicate column index in row 0
        Matrix<TConfig> A_dup_d;
        upload_csr_to_device(2, {0, 3, 5}, {0, 1, 1, 0, 1}, {2.0, -0.25, -0.25, -0.5, 2.0}, A_dup_d);
        expect_setup_failure(A_dup_d, "int4_split");
    }
    {
        // NaN off-diagonal entry must fail cleanly
        Matrix<TConfig> A_nan_d;
        double qnan = std::numeric_limits<double>::quiet_NaN();
        upload_csr_to_device(2, {0, 2, 4}, {0, 1, 0, 1}, {2.0, qnan, qnan, 2.0}, A_nan_d);
        expect_setup_failure(A_nan_d, "int4_split");
    }

    // 4. FP32 runtime range guards (dFFI): Pass-1 off-diagonal subnormal oor fallback & Pass-2 overflow recovery
    if constexpr (std::is_same_v<typename TConfig::MatPrec, float> &&
                  std::is_same_v<typename TConfig::VecPrec, float>)
    {
        // Pass-1 off-diagonal subnormal underflow: A = [2^-48, -2^-52; -2^-52, 2^-4 + 2^-56], r = [2^-149, 0]
        // Exact G_q^T G_q r has x[1] > 0; must trigger double fallback rather than underflowing to 0.
        Matrix<TConfig> A_sub_d;
        double a00 = std::ldexp(1.0, -48);
        double a01 = -std::ldexp(1.0, -52);
        double a11 = std::ldexp(1.0, -4) + std::ldexp(1.0, -56);
        upload_csr_to_device(2, {0, 2, 4}, {0, 1, 0, 1}, {a00, a01, a01, a11}, A_sub_d);
        Vector<TConfig_h> b_sub_h(2), x_sub_h(2);
        b_sub_h[0] = std::ldexp(1.0f, -149);
        b_sub_h[1] = 0.0f;
        x_sub_h[0] = 0.0f;
        x_sub_h[1] = 0.0f;
        Vector<TConfig> b_sub_d = b_sub_h;
        Vector<TConfig> x_sub_d = x_sub_h;
        AMG_Config cfg;
        std::string pstr = "solver=BITFSAI, bitfsai_mode=int4_split, bitfsai_omega=1.0, max_iters=1";
        UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
        Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
        smoother->setup(A_sub_d, false);
        smoother->solve(b_sub_d, x_sub_d, true);
        x_sub_h = x_sub_d;
        delete smoother;
        UNITTEST_ASSERT_TRUE(std::isfinite(x_sub_h[0]) && std::isfinite(x_sub_h[1]));
        UNITTEST_ASSERT_TRUE(x_sub_h[1] > 0.0f);

        // Pass-2 pre-damping FP32 overflow recovery: A = [2^-8], G = [16], r = 2^120, omega = 0.5 -> x = 2^127
        Matrix<TConfig> A_ovf_d;
        upload_csr_to_device(1, {0, 1}, {0}, {std::ldexp(1.0, -8)}, A_ovf_d);
        Vector<TConfig_h> b_ovf_h(1), x_ovf_h(1);
        b_ovf_h[0] = std::ldexp(1.0f, 120);
        x_ovf_h[0] = 0.0f;
        Vector<TConfig> b_ovf_d = b_ovf_h;
        Vector<TConfig> x_ovf_d = x_ovf_h;
        AMG_Config cfg2;
        std::string pstr2 = "solver=BITFSAI, bitfsai_mode=int4_split, bitfsai_omega=0.5, max_iters=1";
        UNITTEST_ASSERT_TRUE(cfg2.parseParameterString(const_cast<char *>(pstr2.c_str())) == AMGX_OK);
        Solver<TConfig> *sm2 = SolverFactory<TConfig>::allocate(cfg2, "default", "solver");
        sm2->setup(A_ovf_d, false);
        sm2->solve(b_ovf_d, x_ovf_d, true);
        x_ovf_h = x_ovf_d;
        delete sm2;
        float exp_ovf = std::ldexp(1.0f, 127);
        UNITTEST_ASSERT_TRUE(std::isfinite(x_ovf_h[0]));
        UNITTEST_ASSERT_TRUE(std::fabs(x_ovf_h[0] - exp_ovf) / exp_ovf < 1e-5f);
    }

    // 5. Numerical edge cases (FP64 matrix & vector mode)
    if constexpr (std::is_same_v<typename TConfig::MatPrec, double> &&
                  std::is_same_v<typename TConfig::VecPrec, double>)
    {
        // Tiny pivot 1e-28
        run_2x2_case("int4_split", 1e-28, 0.5e-28, 1e-28,
                     1.0, 0.0, (4.0 / 3.0) * 1e28, -(2.0 / 3.0) * 1e28, 0.15);
        // Small valid coupling (2^42, 2^21)
        double a00 = std::ldexp(1.0, 42);
        double a01 = std::ldexp(1.0, 21);
        run_2x2_case("fp4_e2m1", a00, a01, 2.0,
                     0.0, 1.0, -a01 / a00, 1.0, 1e-3);
        // Sub-FP32 FP64 RHS (1e-48) via wide apply
        run_2x2_case("int4_split", 4.0, -1.0, 4.0,
                     1e-48, -2e-48, (2.0 / 15.0) * 1e-48, -(7.0 / 15.0) * 1e-48, 0.15);
        // Ill-conditioned k=7000 triggers stored-energy diagonal Jacobi fallback
        double k = 7000.0;
        double b00 = k * k + 1.0;
        double b01 = k * (k + 1.0) + 1.0;
        double b11 = (k + 1.0) * (k + 1.0) + 1.0;
        run_2x2_case("fp4_e2m1", b00, b01, b11,
                     1.0, 1.0, 1.0 / b00, 1.0 / b11, 1e-4);

        // Pass-2 pre-damping binary64 overflow recovery: A = [1/4], G = [2], r = 2^1022, omega = 0.5 -> x = 2^1023
        {
            Matrix<TConfig> A_f64_ovf;
            upload_csr_to_device(1, {0, 1}, {0}, {0.25}, A_f64_ovf);
            Vector<TConfig_h> b1_h(1), x1_h(1);
            b1_h[0] = std::ldexp(1.0, 1022);
            x1_h[0] = 0.0;
            Vector<TConfig> b1_d = b1_h, x1_d = x1_h;
            AMG_Config cfg;
            std::string pstr = "solver=BITFSAI, bitfsai_mode=int4_split, bitfsai_omega=0.5, max_iters=1";
            UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
            Solver<TConfig> *sm = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
            sm->setup(A_f64_ovf, false);
            sm->solve(b1_d, x1_d, true);
            x1_h = x1_d;
            delete sm;
            UNITTEST_ASSERT_TRUE(std::isfinite(x1_h[0]) && x1_h[0] == std::ldexp(1.0, 1023));
        }

        // Certified safe_omega on A=[3], G=[1]: omega * 3 <= 1.0 strictly
        {
            Matrix<TConfig> A3;
            upload_csr_to_device(1, {0, 1}, {0}, {3.0}, A3);
            Vector<TConfig_h> b1_h(1), x1_h(1);
            b1_h[0] = 1.0; x1_h[0] = 0.0;
            Vector<TConfig> b1_d = b1_h, x1_d = x1_h;
            AMG_Config cfg;
            std::string pstr = "solver=BITFSAI, bitfsai_mode=fp64, bitfsai_omega=1.0, bitfsai_safe_omega=1, max_iters=1";
            UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
            Solver<TConfig> *sm = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
            sm->setup(A3, false);
            sm->solve(b1_d, x1_d, true);
            x1_h = x1_d;
            delete sm;
            // M = 1/3, so x = omega * (1/3) and omega * lambda_max(M*A) = omega <= 1.0
            // Exact U_upper = 1.0 for G = 1/sqrt(3), A = 3 -> omega_cap <= 1.0 -> x <= 1/3
            UNITTEST_ASSERT_TRUE(x1_h[0] > 0.0 && 3.0L * static_cast<long double>(x1_h[0]) <= 1.0L);
        }

        // Unrepresentable 1e100 diagonal must fail cleanly
        Matrix<TConfig> A_huge_d;
        upload_csr_to_device(2, {0, 1, 2}, {0, 1}, {1e100, 1e100}, A_huge_d);
        expect_setup_failure(A_huge_d, "int4_split");

        // Non-positive diagonal must fail cleanly
        Matrix<TConfig> A_neg_d;
        upload_csr_to_device(2, {0, 1, 2}, {0, 1}, {1.0, -2.0}, A_neg_d);
        expect_setup_failure(A_neg_d, "int4_split");

        // Failed same-row-count resetup (2x2 -> rejected 2x3) must leave solver unready
        {
            Matrix<TConfig> A_ok_d;
            upload_csr_to_device(2, {0, 1, 2}, {0, 1}, {2.0, 2.0}, A_ok_d);

            Matrix<TConfig_h> A_rect_h;
            A_rect_h.set_initialized(0);
            A_rect_h.addProps(CSR);
            A_rect_h.resize(2, 3, 2, 1, 1, 0);
            A_rect_h.row_offsets[0] = 0; A_rect_h.row_offsets[1] = 1; A_rect_h.row_offsets[2] = 2;
            A_rect_h.col_indices[0] = 0; A_rect_h.col_indices[1] = 2;
            A_rect_h.values[0] = 2.0; A_rect_h.values[1] = 2.0;
            A_rect_h.set_initialized(1);
            Matrix<TConfig> A_rect_d = A_rect_h;

            AMG_Config cfg;
            std::string pstr = "solver=BITFSAI, bitfsai_mode=int4_split, bitfsai_omega=1.0, max_iters=1";
            UNITTEST_ASSERT_TRUE(cfg.parseParameterString(const_cast<char *>(pstr.c_str())) == AMGX_OK);
            Solver<TConfig> *smoother = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
            UNITTEST_ASSERT_TRUE(smoother != NULL);
            smoother->setup(A_ok_d, false);

            bool setup_threw = false;
            try { smoother->setup(A_rect_d, false); }
            catch (const amgx_exception &) { setup_threw = true; }
            UNITTEST_ASSERT_TRUE(setup_threw);

            Vector<TConfig> b2(2, 1.0), x2(2, 0.0);
            bool solve_threw = false;
            try { smoother->solve(b2, x2, true); }
            catch (const amgx_exception &) { solve_threw = true; }
            delete smoother;
            UNITTEST_ASSERT_TRUE(solve_threw);
        }
    }
}

DECLARE_UNITTEST_END(BitFSAISmoother);

BitFSAISmoother<TemplateMode<AMGX_mode_dDDI>::Type> BitFSAISmoother_instance_mode_dDDI;
BitFSAISmoother<TemplateMode<AMGX_mode_dFFI>::Type> BitFSAISmoother_instance_mode_dFFI;

} // namespace amgx
