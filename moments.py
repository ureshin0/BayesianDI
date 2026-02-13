import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import healpy as hp
import healjax as hj

# Import core physics and matrix functions
from physics import (
    angular_distance, 
    get_full_design_matrix, 
    get_exposure_matrix
)

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def compute_analytical_posterior(w, v, i, u1, u2, mu_a, sigma_a, sigma_d, ell, d, P, 
                                 obs_times, wl, line_profile, nside, dmat):
    """
    Computes the analytical posterior mean and covariance for the surface map 
    given a specific set of MCMC parameters.
    """
    npix = hp.nside2npix(nside)
    
    # Generate the design matrix W for this sample
    W = get_full_design_matrix(nside, v, i, u1, u2, obs_times, P, wl, line_profile, w)
    
    # Construct Prior Covariance Matrix (Gaussian Process with RBF kernel)
    Sigma_a = sigma_a**2 * jnp.exp(-(dmat**2) / (2 * ell**2)) + 1e-6 * jnp.eye(npix)
    Lambda_a = jnp.linalg.inv(Sigma_a)
    
    # Instrumental noise precision
    inv_var_d = 1.0 / (sigma_d**2 + 1e-6)
    
    # Posterior Precision Matrix: Lambda_post = Sigma_a^-1 + W.T @ Sigma_d^-1 @ W
    WT_SigmadInv_W = (W.T * inv_var_d) @ W
    Precision_post = Lambda_a + WT_SigmadInv_W
    Sigma_post = jnp.linalg.inv(Precision_post)
    
    # Posterior Mean: mu_post = mu_a + Sigma_post @ W.T @ Sigma_d^-1 @ (d - W @ mu_a)
    Mu_a_vec = mu_a * jnp.ones(npix)
    residual = d.reshape(-1) - W @ Mu_a_vec
    projected_residual = (W.T * inv_var_d) @ residual
    mu_post = Mu_a_vec + Sigma_post @ projected_residual
    
    return mu_post, Sigma_post

def calculate_posterior_moments(samples, d, obs_times, wl, line_profile, nside, dmat, use_indices=None):
    """
    Calculates the ensemble posterior mean and variance maps by averaging over MCMC samples.
    """
    v_all = jnp.asarray(samples['v'])
    i_all = jnp.arccos(jnp.asarray(samples['cosi']))
    u1_all = jnp.asarray(samples['u1'])
    u2_all = jnp.asarray(samples['u2'])
    w_all = jnp.exp(jnp.asarray(samples['log_w']))
    sigma_d_all = jnp.asarray(samples['sigma_d'])
    mu_a_all = jnp.asarray(samples['mu_a'])
    sigma_a_all = jnp.asarray(samples['sigma_a'])
    ell_all = jnp.asarray(samples['ell'])
    P_all = jnp.asarray(samples['P'])

    if use_indices is None:
        idx_array = np.arange(len(v_all))
    else:
        idx_array = np.asarray(use_indices)
        
    npix = hp.nside2npix(nside)
    mu_list = []
    Sigma_diag_sum = jnp.zeros(npix)

    print(f"Processing {len(idx_array)} samples...")
    for count, idx in enumerate(idx_array):
        mu_s, Sigma_s = compute_analytical_posterior(
            w_all[idx], v_all[idx], i_all[idx], u1_all[idx], u2_all[idx],
            mu_a_all[idx], sigma_a_all[idx], sigma_d_all[idx], ell_all[idx],
            d, P_all[idx], obs_times, wl, line_profile, nside, dmat
        )
        mu_list.append(mu_s)
        Sigma_diag_sum += jnp.diag(Sigma_s)
        
        if (count + 1) % 10 == 0:
            print(f"Sample {count + 1} / {len(idx_array)} completed")

    mu_stack = jnp.stack(mu_list, axis=0)
    mu_bar = jnp.mean(mu_stack, axis=0)
    
    # Law of Total Variance: Var = E[Var(Map|Params)] + Var(E[Map|Params])
    diff2_mean = jnp.mean((mu_stack - mu_bar[None, :]) ** 2, axis=0)
    var_diag = (Sigma_diag_sum / len(idx_array)) + diff2_mean

    return mu_bar, var_diag

def plot_maps(mu_bar, std_diag, i_deg, out_dir, prefix):
    """Generates and saves Mollweide and Orthographic projection plots."""
    os.makedirs(out_dir / "figs", exist_ok=True)
    
    # 1. Mean Map (Mollweide)
    hp.mollview(mu_bar, cmap='inferno', title="Posterior Mean Map", unit="Intensity", flip='geo')
    hp.graticule()
    plt.savefig(out_dir / "figs" / f"{prefix}_mean_moll.png", dpi=200)
    plt.close()

    # 2. Mean Map (Orthographic)
    hp.orthview(mu_bar, rot=(0, 90 - i_deg, 0), cmap='inferno', title="Posterior Mean Map", flip='geo')
    hp.graticule()
    plt.savefig(out_dir / "figs" / f"{prefix}_mean_orth.png", dpi=200)
    plt.close()

    # 3. Uncertainty Map (Mollweide)
    hp.mollview(std_diag, cmap='viridis', title="Posterior Std Dev", unit="Intensity", flip='geo')
    hp.graticule()
    plt.savefig(out_dir / "figs" / f"{prefix}_std_moll.png", dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze DI MCMC samples and generate maps')
    parser.add_argument('--chip', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--res_dir', type=str, default='./results')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use for mean map')
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    res_path = Path(args.res_dir)
    
    # 1. Load Data and Samples
    # (Same loading logic as estimate.py)
    pp = np.load(data_path / "posterior_predictive_vsini=0.npz")
    with open(data_path / "fainterspectral-fits_6.pickle", "rb") as f:
        cr = pickle.load(f, encoding="latin1")

    chip_idx = args.chip
    wl = cr["wobs"][chip_idx] * 1e4
    sort_idx = np.argsort(wl)
    wl = wl[sort_idx]
    d = (cr["obs1"] / cr["chipcors"])[:, chip_idx, sort_idx]
    line_profile = np.interp(wl, pp["wav"], pp["mu_med"])

    sample_file = res_path / f"mcmc_chip{chip_idx}_results.npz"
    samples = np.load(sample_file)

    # 2. Setup Geometry
    obs_times = jnp.array([0.1447, 0.5291, 0.9135, 1.2987, 1.6831, 2.0676, 2.4526, 
                           2.8374, 3.2220, 3.6075, 3.9924, 4.3760, 4.7607, 5.1447])
    nside = 8
    npix = hp.nside2npix(nside)
    vmap_pix2ang = jax.vmap(partial(hj.pix2ang, 'ring', nside))
    th, ph = vmap_pix2ang(jnp.arange(npix))
    dmat = angular_distance(th[:, None], ph[:, None], th[None, :], ph[None, :])

    # 3. Compute Posterior Moments
    # Using a subset of samples for efficiency
    subset_indices = np.linspace(0, len(samples['v'])-1, args.num_samples, dtype=int)
    mu_bar, var_diag = calculate_posterior_moments(
        samples, d, obs_times, wl, line_profile, nside, dmat, use_indices=None
    )

    # 4. Save and Plot
    np.save(res_path / f"posterior_mean_chip{chip_idx}.npy", mu_bar)
    np.save(res_path / f"posterior_var_chip{chip_idx}.npy", var_diag)
    
    i_mean_deg = np.rad2deg(np.arccos(np.mean(samples['cosi'])))
    plot_maps(np.asarray(mu_bar), np.sqrt(np.asarray(var_diag)), i_mean_deg, res_path, f"chip{chip_idx}")
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()