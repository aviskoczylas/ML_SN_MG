import time
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from numpy.polynomial.legendre import leggauss
from scipy.special import lpmv
from numba import njit, prange

@njit(fastmath=True)
def _half_moment(leg, w, psi):
    L = leg.shape[0]
    Nh = w.size
    out = np.zeros(L)
    for l in range(L):
        acc = 0.0
        for n in range(Nh):
            acc += leg[l, n] * w[n] * psi[n]
        out[l] = acc
    return out

@njit(fastmath=True)
def _sweep_sc_one_group(sigt_g, dx, mu_pos, mu_neg, src, leg_pos, leg_neg, w_pos, w_neg,
                        bc_type, left_in_pos, right_in_neg):
    S = sigt_g.size
    L = leg_pos.shape[0]
    Nh = mu_pos.size
    mom = np.zeros((S, L))

    psi_in = np.zeros(Nh)
    for j in range(Nh):
        psi_in[j] = 0.0 if bc_type != 1 else left_in_pos[j]

    for i in range(S):
        sigt = sigt_g[i]
        q    = src[i, :Nh]
        tau  = sigt * dx / mu_pos
        e    = np.exp(-tau)

        psi_out = psi_in * e + (q / sigt) * (1.0 - e)

        t1 = np.empty_like(tau)
        for n in range(Nh):
            t1[n] = (1.0 - e[n]) / tau[n] if tau[n] > 1e-12 else (1.0 - 0.5 * tau[n])
        psi_bar = psi_in * t1 + (q / sigt) * (1.0 - t1)

        mom[i, :] += _half_moment(leg_pos, w_pos, psi_bar)
        psi_in = psi_out

    if bc_type == 0:
        psi_in = psi_out[::-1]
    elif bc_type == 1:
        psi_in = right_in_neg.copy()
    else:
        psi_in = np.zeros(Nh)

    for i in range(S - 1, -1, -1):
        sigt = sigt_g[i]
        q    = src[i, Nh:]
        tau  = sigt * dx / mu_neg
        e    = np.exp(-tau)

        psi_out = psi_in * e + (q / sigt) * (1.0 - e)

        t1 = np.empty_like(tau)
        for n in range(Nh):
            t1[n] = (1.0 - e[n]) / tau[n] if tau[n] > 1e-12 else (1.0 - 0.5 * tau[n])
        psi_bar = psi_in * t1 + (q / sigt) * (1.0 - t1)

        mom[i, :] += _half_moment(leg_neg, w_neg, psi_bar)
        psi_in = psi_out

    return mom

@njit(parallel=True, fastmath=True)
def _outer_iteration_parallel(flux_prev, sigt_sec, scatter_mats_sec, sigma_f, chi_sec,
                              prod_mask, dx, mu, w, leg_full,
                              bc_type, left_in_pos, right_in_neg,
                              theta_relax, max_inner, eps_inner,nu_over_k):
    S, L, G = flux_prev.shape
    N = mu.size
    Nh = N // 2

    mu_pos = np.abs(mu[:Nh]); mu_neg = np.abs(mu[Nh:])
    w_pos  = w[:Nh];           w_neg  = w[Nh:]
    leg_pos = leg_full[:, :Nh]
    leg_neg = leg_full[:, Nh:]

    prod_rate = np.zeros(S)
    for s in prange(S):
        acc = 0.0
        for g in range(G):
            acc += sigma_f[g] * flux_prev[s, 0, g]
        prod_rate[s] = nu_over_k * acc * prod_mask[s]

    g2g_amp = np.zeros((S, G, L))
    for s in prange(S):
        for g in range(G):
            for l in range(L):
                acc = 0.0
                for gp in range(G):
                    if gp != g:
                        acc += scatter_mats_sec[l, s, gp, g] * flux_prev[s, l, gp]
                g2g_amp[s, g, l] = acc

    sig_diag = np.zeros((L, S, G))
    for l in prange(L):
        for s in range(S):
            for g in range(G):
                sig_diag[l, s, g] = scatter_mats_sec[l, s, g, g]

    flux_next = np.zeros_like(flux_prev)

    for g in prange(G):
        group_mom = flux_prev[:, :, g].copy()
        prev_mom  = group_mom.copy()

        for _ in range(max_inner):
            total_src = np.zeros((S, N))

            for s in range(S): #add fission source
                amp = prod_rate[s] * chi_sec[s, g]
                for n in range(N):
                    total_src[s, n] += amp

            for l in range(L):
                coeff = 0.5 * (2 * l + 1)
                for s in range(S):
                    amp = coeff * (sig_diag[l, s, g] * group_mom[s, l] + g2g_amp[s, g, l])
                    for n in range(N):
                        total_src[s, n] += amp * leg_full[l, n]

            sigt_g = sigt_sec[:, g]
            new_mom = _sweep_sc_one_group(sigt_g, dx, mu_pos, mu_neg, total_src,
                                          leg_pos, leg_neg, w_pos, w_neg,
                                          bc_type, left_in_pos, right_in_neg)

            num = 0.0; den = 0.0
            for s in range(S):
                for l in range(L):
                    d = new_mom[s, l] - prev_mom[s, l]
                    num += d * d
                    den += prev_mom[s, l] * prev_mom[s, l]
            l2 = np.sqrt(num / (den + 1e-30))

            for s in range(S):
                for l in range(L):
                    group_mom[s, l] = theta_relax * new_mom[s, l] + (1.0 - theta_relax) * group_mom[s, l]
            prev_mom[:, :] = new_mom

            if l2 < eps_inner:
                break

        flux_next[:, :, g] = group_mom

    fsum = 0.0
    for s in range(S):
        fsum += prod_rate[s]
        
    return flux_next, fsum

class Sn:
    def __init__(self, num_ords, num_groups, num_nodes, NH, dx, bc_type="reflecting"):
        self.num_ordinates = int(num_ords)
        assert self.num_ordinates % 2 == 0
        self.num_groups = int(num_groups)
        self.num_nodes = int(num_nodes)
        self.dx = float(dx)
        self.tol = 1e-6

        assert abs(1.0/self.dx - int(1.0/self.dx)) < self.tol
        self.sections_per_cell = int(round(1.0/self.dx))
        self.num_sections = self.num_nodes * self.sections_per_cell
        self.section_index_to_cell = (np.arange(self.num_sections) // self.sections_per_cell).astype(int)

        self.AH, self.AU = 1, 238
        self.NH, self.NU = float(NH), 1.0

        self.E0, self.Emin = 1e6, 1.0
        self.nbins = self.num_groups + 1
        self.boundaries = None
        self.Evec = None

        self.nu = 2.43
        self.bc_type = bc_type
        self.data_dir = "./data/"

        self.leg_order = 4
        self.legendre = np.zeros((self.leg_order, self.num_ordinates))
        x, w = leggauss(self.num_ordinates)
        self.weight = np.flip(w)
        self.mu = np.flip(x)
        self._build_legendre_matrix()

        self.sig_t_H = None; self.sig_s0_H = None
        self.sig_t_U = None; self.sig_s0_U = None
        self.sigma_f = None
        self.chi = None

        self.xsH_gtg = None
        self.xsU_gtg = None

        self.sigt_sec = None
        self.scatter_mats_sec = None

        self._load_or_build_data()
        self.build_geometry(seed=10, p_U=0.8, verbose=True)

    @staticmethod
    def _safe_load_csv(path, sep, cols=None):
        df = pd.read_csv(path, sep=sep, header=0)
        if cols is not None:
            try:
                df = df.loc[:, cols]
            except Exception:
                pass
        return df.to_numpy(dtype=float)

    @staticmethod
    def get_data(data, gridpoints, E0, Emin):
        data = data[(data[:, 0] <= E0) & (data[:, 0] >= Emin)].copy()
        data[:, 0] = np.log(E0 / data[:, 0])
        data = data[np.argsort(data[:, 0])]
        new_grid = np.linspace(np.log(E0/E0), np.log(E0/Emin), gridpoints)
        out = np.empty((new_grid.size, data.shape[1]), dtype=float)
        out[:, 0] = new_grid
        xp = data[:, 0]
        for i in range(1, data.shape[1]):
            out[:, i] = np.interp(new_grid, xp, data[:, i])
        return out

    def _load_or_build_data(self):
        chi35 = self._safe_load_csv(f'{self.data_dir}chi_u235.txt', '\t', cols=['E','chi'])
        H1   = self._safe_load_csv(f'{self.data_dir}xs_h1_T293k.txt', '\t', cols=['E','sigma_t','sigma_s'])
        U238 = self._safe_load_csv(f'{self.data_dir}xs_u238_T293k.txt', '\t', cols=['E','sigma_t','sigma_s'])
        sigma_f_raw = self._safe_load_csv(f'{self.data_dir}xs_u238_fission.csv', ',', cols=None)

        chi = self.get_data(chi35, self.nbins, self.E0, self.Emin)
        H   = self.get_data(H1,   self.nbins, self.E0, self.Emin)
        U   = self.get_data(U238, self.nbins, self.E0, self.Emin)

        self.boundaries = U[:, 0]
        self.Evec = self.E0 * np.exp(-self.boundaries)

        self.sig_t_H  = H[:-1, 1] * self.NH
        self.sig_s0_H = H[:-1, 2] * self.NH
        self.sig_t_U  = U[:-1, 1] * self.NU
        self.sig_s0_U = U[:-1, 2] * self.NU

        self.chi = chi[:-1, 1]
        s = self.chi.sum()
        if s > 0: self.chi /= s

        if sigma_f_raw is not None and sigma_f_raw.shape[1] >= 2:
            self.sigma_f = 5 * (self.get_data(sigma_f_raw, self.nbins, self.E0, self.Emin)[:-1, 1])
        else: self.sigma_f = np.zeros(self.num_groups)

        self.xsH_gtg = self._build_sigma_gtg(self.AH, self.sig_s0_H)
        self.xsU_gtg = self._build_sigma_gtg(self.AU, self.sig_s0_U)

    def _build_sigma_gtg(self, A, sig_s0):
        gmax_vec = self._gmax_vec_fn(A, self._lga_fn(self._alpha_fn(A)))
        return self._gen_sig_sn_gtg(A, sig_s0, self.leg_order, self.boundaries, gmax_vec,
                                    self._alpha_fn(A), self.tol)

    def build_geometry(self, layout=None, p_U=0.5, seed=10, verbose=True):
        if layout is not None:
            arr = np.asarray(layout, dtype=int)
            assert arr.shape == (self.num_nodes,)
            assert np.all((arr == 0) | (arr == 1))
            self.cell_layout = arr.copy()
        else:
            rng = np.random.default_rng(seed=seed)
            self.cell_layout = (rng.random(self.num_nodes) < float(p_U)).astype(int)

        self.mat_per_section = np.repeat(self.cell_layout, self.sections_per_cell)
        if verbose:
            print(f"Geometry (cells) 0=H,1=U: {self.cell_layout}")
            print(f"Number of Uranium Cells: {sum(self.cell_layout)}")
        self._build_per_section_xs_from_layout()

    def _build_per_section_xs_from_layout(self):
        S, G = self.num_sections, self.num_groups
        self.sigt_sec = np.zeros((S, G))
        self.scatter_mats_sec = np.zeros((self.leg_order, S, G, G))
        for s in range(S):
            if self.mat_per_section[s] == 0:
                self.sigt_sec[s, :] = self.sig_t_H
                self.scatter_mats_sec[:, s, :, :] = self.xsH_gtg
            else:
                self.sigt_sec[s, :] = self.sig_t_U
                self.scatter_mats_sec[:, s, :, :] = self.xsU_gtg

    def _build_legendre_matrix(self):
        for l in range(self.leg_order):
            self.legendre[l, :] = lpmv(0, l, self.mu)

    @staticmethod
    def _alpha_fn(A): return ((A - 1.0) / (A + 1.0)) ** 2
    @staticmethod
    def _lga_fn(alpha): return -np.log(alpha) if alpha != 0 else np.inf

    def _group_bound(self, A, g, lga):
        if A == 1.0:
            return self.boundaries.size
        return int(np.searchsorted(self.boundaries, self.boundaries[g] + lga))

    def _gmax_vec_fn(self, A, lga):
        G = self.boundaries.size
        gmax_vec = np.zeros(G, dtype=np.int64)
        for g in range(G):
            gmax_vec[g] = self._group_bound(A, g, lga)
        return gmax_vec

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _gen_sig_sn_gtg(A, sig_s0, order, boundaries, gmax_vec, alpha, tol, n_sub=8):
        G = boundaries.size
        du = boundaries[1] - boundaries[0]
        den = (1 - alpha) * du
        lga = -np.log(alpha) if A != 1 else np.inf
        sigma_gtg = np.zeros((order, G - 1, G - 1))
        Am1 = A - 1.0
        Ap1 = A + 1.0

        for l in range(order):
            for gp in prange(G - 1):
                x1 = boundaries[gp]
                x2 = boundaries[gp + 1]

                gmax = gmax_vec[gp] if gmax_vec[gp] < G else (G - 1)
                for g in range(gp, gmax):
                    y1 = boundaries[g]
                    y2 = boundaries[g + 1]
                    c = x1 if x1 > (y1 - lga) else (y1 - lga)

                    if (y1 < x1) or (c >= x2) or den == 0.0:
                        continue

                    length = x2 - c
                    n_steps_base = int(np.ceil(length / du))
                    if n_steps_base < 1:
                        n_steps_base = 1
                    n_steps = n_steps_base * n_sub
                    dx = length / n_steps

                    acc = 0.0
                    for s in range(n_steps):
                        xm = c + (s + 0.5) * dx
                        a = y1 if xm < y1 else xm
                        bx = xm + lga
                        b = y2 if bx > y2 else bx

                        if l == 0:
                            val = np.exp(-(a - xm)) - np.exp(-(b - xm))
                        elif l == 1:
                            if A == 1.0:
                                val = (Ap1 / 3.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                            else:
                                val = ((Ap1 / 3.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                                       - (Am1) * (np.exp(0.5 * (xm - a)) - np.exp(0.5 * (xm - b))))
                        elif l == 2:
                            if A == 1.0:
                                val = 0.0625 * ((np.exp(xm - 2 * a - b) - np.exp(xm - a - 2 * b)) *
                                                (-4.0 * (3.0 * A * A - 1.0) * np.exp(a + b)
                                                 + (3.0 * Ap1 * Ap1 * (np.exp(a + xm) + np.exp(b + xm)))))
                            else:
                                val = 0.0625 * (6.0 * Am1 * Am1 * (b - a)
                                                + (np.exp(xm - 2 * a - b) - np.exp(xm - a - 2 * b)) *
                                                (-4.0 * (3.0 * A * A - 1.0) * np.exp(a + b)
                                                 + (3.0 * Ap1 * Ap1 * (np.exp(a + xm) + np.exp(b + xm)))))
                        else:
                            if A == 1.0:
                                val = 0.0625 * (
                                    2.0 * (Ap1 * Ap1 * Ap1) * (np.exp(2.5 * (xm - a)) - np.exp(2.5 * (xm - b)))
                                    - 2.0 * (Ap1) * (5.0 * A * A - 1.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                                )
                            else:
                                val = 0.0625 * (
                                    10.0 * (Am1 * Am1 * Am1) * (np.exp(0.5 * (a - xm)) - np.exp(0.5 * (b - xm)))
                                    + 2.0 * (Ap1 * Ap1 * Ap1) * (np.exp(2.5 * (xm - a)) - np.exp(2.5 * (xm - b)))
                                    + 6.0 * (Am1) * (5.0 * A * A - 1.0) * (np.exp(0.5 * (xm - a)) - np.exp(0.5 * (xm - b)))
                                    - 2.0 * (Ap1) * (5.0 * A * A - 1.0) * (np.exp(1.5 * (xm - a)) - np.exp(1.5 * (xm - b)))
                                )

                        acc += val

                    sigma_gtg[l, gp, g] = (sig_s0[gp] * acc * dx) / den
                    if abs(sigma_gtg[l, gp, g]) < tol:
                        break

        return sigma_gtg

    def moments_to_flux(self, flux_moments):
        L = self.leg_order
        coeff = (2*np.arange(L)+1)/2.0
        return (flux_moments * coeff[None, :]) @ self.legendre

    def flux_to_moments(self, phi, direction):
        if direction == 1:
            leg = self.legendre[:, :self.num_ordinates//2]
            w = self.weight[:self.num_ordinates//2]
        else:
            leg = self.legendre[:, self.num_ordinates//2:]
            w = self.weight[self.num_ordinates//2:]
        return leg @ (w * phi)

    def _build_chi_per_section(self):
        if not hasattr(self, "mat_per_section"):
            raise RuntimeError("Call build_geometry() first.")
    
        S, G = self.num_sections, self.num_groups
        chi_sec = np.zeros((S, G), dtype=np.float64)
    
        chi_vec = np.array(self.chi, dtype=np.float64).ravel()
        if chi_vec.size != G or chi_vec.sum() <= 0:
            chi_vec = np.zeros(G, dtype=np.float64); chi_vec[0] = 1.0
        else:
            chi_vec /= chi_vec.sum()
    
        chi_sec[self.mat_per_section == 1, :] = chi_vec[None, :]
        return chi_sec

    def in_group_scatter_expansion(self, group_flux_moments, g):
        S, N = self.num_sections, self.num_ordinates
        src = np.zeros((S, N))
        for l in range(self.leg_order):
            sig_ll = self.scatter_mats_sec[l, :, g, g]
            term = ((2*l+1)/2.0) * (sig_ll * group_flux_moments[:, l])
            src += term[:, None] * self.legendre[l, :][None, :]
        return src

    def g2g_scatter_expansion(self, flux_moments, g):
        S, N = self.num_sections, self.num_ordinates
        src = np.zeros((S, N))
        for l in range(self.leg_order):
            sig_mat = self.scatter_mats_sec[l, :, :, :]
            col_g = sig_mat[:, :, g]
            contrib = np.sum(col_g * flux_moments[:, l, :], axis=1) - col_g[:, g] * flux_moments[:, l, g]
            src += ((2*l+1)/2.0) * contrib[:, None] * self.legendre[l, :][None, :]
        return src

    @staticmethod
    def converged(prev, curr, label=None, eps=1e-6):
        if prev is None:
            return False
        diff = np.linalg.norm(curr - prev)
        norm = np.linalg.norm(prev)
        l2 = diff / (norm + 1e-12)
        if label:
            print(f"{label} L2: {l2:.3e}")
        return l2 < eps

    def run_parallel(self, max_outer=40, alpha_underrelax=0.7, eps_outer=1e-6,
                     theta_relax=0.6, max_inner=60, eps_inner=5e-4):
        S, G, L = self.num_sections, self.num_groups, self.leg_order
        N = self.num_ordinates

        chi_sec = self._build_chi_per_section()
        prod_mask = (self.mat_per_section == 1).astype(np.float64)

        Nh = N // 2
        if self.bc_type == "fixed":
            m0 = 2.0  # try 1~5 to test sensitivity; ψ_in ≈ m0/2
            left_in  = self.moments_to_flux(np.asarray([m0] + [0.0]*(L-1))[None, :])[0, :Nh]
            right_in = self.moments_to_flux(np.asarray([m0] + [0.0]*(L-1))[None, :])[0, Nh:]
        else:
            left_in = np.zeros(Nh)
            right_in = np.zeros(Nh)

        bc_map = {"reflecting": 0, "fixed": 1, "vacuum": 2}
        bc_code = bc_map[self.bc_type]

        sigt_sec = self.sigt_sec.astype(np.float64)
        scat = self.scatter_mats_sec.astype(np.float64)
        leg_full = self.legendre.astype(np.float64)
        mu = self.mu.astype(np.float64)
        w = self.weight.astype(np.float64)
        chi_sec = chi_sec.astype(np.float64)
        prod_mask = prod_mask.astype(np.float64)
        sigma_f = self.sigma_f.astype(np.float64)
        dx = float(self.dx)

        flux = np.zeros((S, L, G), dtype=np.float64)
        flux[:, 0, :] = 1e-2

        k = 1.0
        prev_fsum = None

        for it in range(max_outer):
            nu_over_k = self.nu / max(k, 1e-30)
            flux_new, fsum = _outer_iteration_parallel(
                flux, sigt_sec, scat, sigma_f, chi_sec, prod_mask, dx, mu, w, leg_full,
                bc_code, left_in, right_in, theta_relax, max_inner, eps_inner, nu_over_k
            )

            if prev_fsum is not None and prev_fsum > 0.0:
                k_new = k * (fsum / prev_fsum)
                k = alpha_underrelax * k_new + (1.0 - alpha_underrelax) * k
            prev_fsum = fsum

            num = np.linalg.norm(flux_new - flux)
            den = np.linalg.norm(flux) + 1e-30
            l2 = num / den
            print(f"[outer {it}] L2={l2:.3e}, k≈{k:.6f}")
            flux = flux_new
            if l2 < eps_outer:
                break

        return flux, k

# run
num_ordinates = 32
num_groups = 8
num_nodes = 64
NH = 5
dx = 0.01

sn = Sn(num_ordinates, num_groups, num_nodes, NH, dx, bc_type="reflecting")
flux_moments, k_eff = sn.run_parallel()
scalar_flux = flux_moments[:, 0, :]
print("k_eff ~", k_eff)
avg_phi_g = scalar_flux.mean(axis=0)
print("Avg scalar flux per group:", np.array2string(avg_phi_g, precision=4))
print("Elapsed:", time.time() - start_time, "s")

S, N, G = sn.num_sections, sn.num_ordinates, sn.num_groups
scalar_flux = flux_moments[:, 0, :]                   # (S, G)
mat = sn.mat_per_section  # 0=H(A=1), 1=U(A=238)

def shade_material(ax):
    C = sn.num_nodes
    cells = sn.cell_layout  # 0=H, 1=U per node
    i = 0
    while i < C:
        m = cells[i]
        j = i + 1
        while j < C and cells[j] == m:
            j += 1
        ax.axvspan(i - 0.5, j - 0.5, color=('blue' if m == 0 else 'red'),
                   alpha=0.12, linewidth=0)
        i = j

save_dir = "charts"
for g in range(G):
    # Angular flux colorplot
    ang = sn.moments_to_flux(flux_moments[:, :, g])  
    fig, ax = plt.subplots(figsize=(6, 3.8))
    im = ax.imshow(ang.T, aspect='auto', origin='lower',
                   extent=[0, sn.num_nodes, -1, 1], interpolation='nearest')
    ax.set_title(f"Group {g} — Angular flux")
    ax.set_xlabel("Section")
    ax.set_ylabel("μ")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("ψ")
    fig.savefig(f"{save_dir}/psi_g{g}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Scalar flux scatter with H/U shading
    x = sn.section_index_to_cell
    fig, ax = plt.subplots(figsize=(6, 3.2))
    shade_material(ax)
    ax.plot(x, scalar_flux[:, g])
    ax.set_title(f"Group {g} — Scalar flux")
    ax.set_xlabel("Section")
    ax.set_ylabel("ϕ₀")
    # optional legend
    legend_patches = [Patch(facecolor='blue', alpha=0.12, label='A=1 (H)'),
                      Patch(facecolor='red',  alpha=0.12, label='A=238 (U)')]
    ax.legend(handles=legend_patches, loc='upper right', frameon=False)
    fig.savefig(f"{save_dir}/phi_g{g}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

