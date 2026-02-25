import os
from pathlib import Path
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import healjax as hp

# Import core physics functions from our local module
from src.physics import angular_distance, get_full_design_matrix

# Enable 64-bit precision for scientific computing
jax.config.update("jax_enable_x64", True)

def run_mcmc_estimation(d, wl, line_profile, obs_times, dmat, nside, key_seed=0):
    """
    Sets up and runs the NumPyro MCMC to estimate stellar parameters.
    """
    n_phase = d.shape[0]
    n_wav = len(wl)
    key = jax.random.PRNGKey(key_seed)

    def model_viu(data):
        # Priors for stellar parameters
        cosi = numpyro.sample('cosi', dist.Uniform(0, 1))
        i = jnp.arccos(cosi) #
        
        # Rotational velocity (km/s)
        v = numpyro.sample('v', dist.Uniform(0.0, 120.0)) #
        #v = numpyro.sample('v', dist.Normal(30.0, 10.0)) #

        # Quadratic Limb Darkening parameters (Kipping 2013 sampling)
        q1 = numpyro.sample('q1', dist.Uniform(0.0, 1.0))
        q2 = numpyro.sample('q2', dist.Uniform(0.0, 1.0))
        sqrt_q1 = jnp.sqrt(q1)
        u1 = numpyro.deterministic('u1', 2.0 * sqrt_q1 * q2) #
        u2 = numpyro.deterministic('u2', sqrt_q1 * (1.0 - 2.0 * q2)) #

        # Phase-dependent scaling factors (instrumental/normalization corrections)
        log_w = numpyro.sample("log_w", dist.Normal(0.0, 0.1).expand([n_phase]))
        w = jnp.exp(log_w) #

        # Rotation Period (days)
        P = numpyro.deterministic('P', 5.0) #

        # Generate the full design matrix W using the physics module
        Wviu = get_full_design_matrix(nside, v, i, u1, u2, obs_times, P, wl, line_profile, w)

        # Noise parameters
        sigma_d = numpyro.sample('sigma_d', dist.LogNormal(jnp.log(0.03), 1.0))
        cov_diag = sigma_d**2 * jnp.ones(Wviu.shape[0]) + 1e-6 

        # Surface map Gaussian Process priors
        mu_a = numpyro.sample('mu_a', dist.Uniform(0.0, 0.05))
        Mu_a = mu_a * jnp.ones(Wviu.shape[1])
        
        sigma_a = numpyro.sample('sigma_a', dist.HalfNormal(0.3))
        ell = numpyro.sample('ell', dist.Uniform(0.1, 1.5))
        
        # Radial Basis Function (RBF) Kernel on the sphere
        Sigma_a = sigma_a**2 * jnp.exp(-(dmat**2) / (2 * ell**2))

        # Efficient likelihood calculation using LowRankMultivariateNormal
        n_dimension = Sigma_a.shape[0]
        jitter = 0.5e-6
        L_a = jnp.linalg.cholesky(Sigma_a + jitter * jnp.eye(n_dimension))
        cov_factor = Wviu @ L_a

        numpyro.sample('obs', dist.LowRankMultivariateNormal(
            loc=Wviu @ Mu_a, 
            cov_factor=cov_factor, 
            cov_diag=cov_diag
        ), obs=data.reshape(-1)) #

    # Initial values for the sampler
    init_vals = {
        'cosi': 0.5,
        'v': 30.0,
        'q1': 0.5,
        'q2': 0.5,
        'sigma_d': 0.01
    }

    kernel = NUTS(model_viu,
                 target_accept_prob=0.9,
                 dense_mass=True,
                 max_tree_depth=10,
                 init_strategy=init_to_value(values=init_vals))
    
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(key, d)
    mcmc.print_summary()
    
    return mcmc.get_samples()

def main():
    parser = argparse.ArgumentParser(description='Doppler Imaging Stellar Parameter Estimation')
    parser.add_argument('--chip', type=int, default=1, help='Spectral chip index (default: 1)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing input data')
    parser.add_argument('--out_dir', type=str, default='./results', help='Directory to save output')
    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data_dir)
    output_path = Path(args.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Precomputed Posterior Predictive Data
    pp_file = data_path / "posterior_predictive_vsini=0.npz"
    if not pp_file.exists():
        raise FileNotFoundError(f"Missing data: {pp_file}")
    
    pp = np.load(pp_file)
    wav_pp = pp["wav"]
    mu_med = pp["mu_med"]

    # 2. Load Observed DI Spectra
    pickle_file = data_path / "fainterspectral-fits_6.pickle"
    with open(pickle_file, "rb") as f:
        cr = pickle.load(f, encoding="latin1")

    # Data preprocessing
    chip_idx = args.chip
    observed_DI = cr["obs1"] / cr["chipcors"]
    wav_di_chip = cr["wobs"][chip_idx] * 1e4
    sort_idx = np.argsort(wav_di_chip)
    
    wl = wav_di_chip[sort_idx]
    flux_di = observed_DI[:, chip_idx, sort_idx]

    # Interpolate template profile to observed wavelength grid
    line_profile = np.interp(wl, wav_pp, mu_med)

    # 3. Setup Grid and Geometry
    obs_times = jnp.array([
        0.1447, 0.5291, 0.9135, 1.2987, 1.6831, 2.0676, 2.4526, 
        2.8374, 3.2220, 3.6075, 3.9924, 4.3760, 4.7607, 5.1447
    ])
    
    nside = 8
    npix = hp.nside2npix(nside)
    vmap_func = jax.vmap(partial(hp.pix2ang, 'ring', nside))
    th, ph = vmap_func(jnp.arange(npix))
    dmat = angular_distance(th[:, None], ph[:, None], th[None, :], ph[None, :]) #

    # 4. Execute MCMC
    samples = run_mcmc_estimation(flux_di, wl, line_profile, obs_times, dmat, nside)

    # 5. Save Results
    results_file = output_path / f"mcmc_chip{chip_idx}_results_P=5hr.npz"
    save_data = {k: np.asarray(v) for k, v in samples.items()}
    np.savez(results_file, **save_data)
    
    print(f"Estimation complete. Samples saved to {results_file}")

if __name__ == "__main__":
    main()
