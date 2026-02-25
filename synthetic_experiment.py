import jax.numpy as jnp
import numpy as np
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from functools import partial
import healjax as hp
from src.const import c0
import matplotlib.pyplot as plt
import healpy
from src.physics import incline, doppler_shift, limb_darkening
from map import map1, map2, map3
from scipy.stats import norm
from pathlib import Path

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

def get_exposure_matrix(nside, vrot, inclination, u, phase, wavelengths, line_profile):
    """
    Compute the mapping matrix M for a single observational phase.
    
    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    vrot : float
        Equatorial rotation velocity.
    inclination : float
        Stellar inclination in radians.
    u : float
        Limb darkening coefficient.
    phase : float
        Rotational phase.
    wavelengths : jnp.ndarray
        Observed wavelength grid.
    line_profile : jnp.ndarray
        Intrinsic local line profile (template).
    """
    npix = hp.nside2npix(nside) #
    vmap_func = jax.vmap(partial(hp.pix2ang, 'ring', nside))
    theta0, phi0 = vmap_func(jnp.arange(npix)) #
    
    # Update longitudinal phase
    phi_rot = phi0 + phase * 2 * jnp.pi #
    
    # Rotate to observer's frame
    theta, phi = incline(theta0, phi_rot, jnp.pi/2 - inclination) #

    # Calculate line-of-sight velocity and Doppler factor
    vlos = vrot * jnp.cos(jnp.pi/2 - inclination) * jnp.sin(theta0) * jnp.sin(phi_rot) #
    D = doppler_shift(vlos) #
    
    # Interpolate local line profile onto shifted wavelengths
    # Shape: (N_wav, N_pix)
    interped = jnp.interp(wavelengths[:, None] / D, wavelengths, line_profile) #

    # Compute limb darkening and visibility weight
    mu = jnp.sin(theta) * jnp.cos(phi) #
    ld = limb_darkening(u, mu) #

    visible = ((-jnp.pi/2 < phi) & (phi < jnp.pi/2)).astype(float) #
    weight = visible * jnp.sin(theta) * jnp.cos(phi) #

    # Design matrix M for this specific phase
    M = weight[None, :] * ld[None, :] * interped #
    return M

def get_full_design_matrix(nside, vrot, inclination, u, n_phase, wavelengths, line_profile, w):
    """
    Compute the full concatenated design matrix W for all observation phases.
    """
    phases = jnp.arange(n_phase) / n_phase

    def scaled(phase, wk):
        M = get_exposure_matrix(nside, vrot, inclination, u, phase, wavelengths, line_profile)
        return wk * M # Apply scaling factor w_k

    # Vectorize across phases and scale factors
    M_stack = jax.vmap(scaled)(phases, w)

    # Reshape to (N_phase * N_wav, N_pix)
    W = M_stack.reshape((n_phase * len(wavelengths), -1))
    return W

def angular_distance(th1, ph1, th2, ph2):
    """
    Calculate the great-circle distance between two points on a sphere.
    Used for Gaussian Process kernel construction.
    """
    cosg = jnp.cos(th1) * jnp.cos(th2) + jnp.sin(th1) * jnp.sin(th2) * jnp.cos(ph1 - ph2)
    cosg = jnp.clip(cosg, -1.0, 1.0) #
    return jnp.arccos(cosg) #

def generate(true_map, wl, line_profile, true_v, true_i, true_u, true_w, n_phase, nside, key_seed=0):
    """
    Generate synthetic data vector d for a given set of true parameters and surface map.
    
    Parameters
    ----------
    true_map : array-like, shape (npix,)
        The true surface map (intensity values on HEALPix grid).
    wl : array-like, shape (n_wav,)
        Wavelength grid of the observations.
    line_profile : array-like, shape (n_wav,)
        Intrinsic local line profile (template).
    true_v : float
        True equatorial rotation velocity (km/s).
    true_i : float
        True stellar inclination (radians).
    true_u : float
        True limb darkening coefficient.
    true_w : array-like, shape (n_phase,)
        True phase-dependent scaling factors (instrumental/normalization corrections).
    n_phase : int
        Number of observational phases.
    nside : int
        HEALPix resolution parameter.
    """
    true_W = get_full_design_matrix(nside, true_v, true_i, true_u, n_phase, wl, line_profile, true_w)

    d = true_W @ true_map

    # Add noise
    sigma = 0.02 * jnp.max(jnp.abs(d))
    key = jax.random.PRNGKey(key_seed)
    d_noisy = d + jax.random.normal(key, d.shape) * sigma

    return d_noisy

def run_mcmc_estimation(d, wl, line_profile, dmat, n_phase, nside, n_wav, key_seed=0):
    """
    Sets up and runs the NumPyro MCMC to estimate stellar parameters.
    """
    key = jax.random.PRNGKey(key_seed)
    
    def model_viu(data):
        # Priors for stellar parameters
        cosi = numpyro.sample('cosi', dist.Uniform(0, 1))
        i = jnp.arccos(cosi)

        # Rotational velocity (km/s)
        v = numpyro.sample('v', dist.Uniform(0.0, 60.0))

        # Limb darkening coefficient
        u = numpyro.sample('u', dist.Uniform(0.0, 1.0))

        # Phase-dependent scaling factors (instrumental/normalization corrections)
        log_w = numpyro.sample("log_w", dist.Normal(0.0, 0.1).expand([n_phase])        )
        w = jnp.exp(log_w)

        # Generate the full design matrix W
        Wviu = get_full_design_matrix(nside, v, i, u, n_phase, wl, line_profile, w)

        # Noise parameters
        sigma_d = numpyro.sample('sigma_d', dist.HalfNormal(10))
        Sigma_d = sigma_d**2 * jnp.eye(n_phase * n_wav)

        # Surface map Gaussian Process priors
        mu_a    = numpyro.sample('mu_a', dist.Beta(2.0, 2.0))
        Mu_a    = mu_a*jnp.ones(Wviu.shape[1])

        sigma_a = numpyro.sample('sigma_a', dist.HalfNormal(0.3))
        log_ell = numpyro.sample('log_ell', dist.Normal(-1.0, 0.5))
        ell     = jnp.exp(log_ell)

        # Radial Basis Function (RBF) Kernel on the sphere
        Sigma_a = sigma_a**2 * jnp.exp(-(dmat**2)/(2*ell**2))

        C = Sigma_d + Wviu @ Sigma_a @ Wviu.T + 1e-6 * jnp.eye(Wviu.shape[0])
        numpyro.sample('obs', dist.MultivariateNormal(loc=Wviu@Mu_a, covariance_matrix=C), obs=data)

    kernel = NUTS(model_viu,
            target_accept_prob=0.9,
            dense_mass=True,      # フル質量行列
            max_tree_depth=8
            )
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(key, d)
    mcmc.print_summary()

    return mcmc.get_samples()

def QR(A,b):
    Q, R = jnp.linalg.qr(A)
    x = jnp.linalg.solve(R, jnp.dot(Q.T, b))
    return x

def mu2(w, v, i, u, mu_a, sigma_a, sigma_d, ell, d, nside, n_phase, n_wav, wl, line_profile, dmat):
    W = get_full_design_matrix(nside, v, i, u, n_phase, wl, line_profile, w)
    Sigma_d = sigma_d**2 * jnp.eye(n_phase * n_wav)
    Sigma_a = sigma_a**2 * jnp.exp(-(dmat**2)/(2*ell**2))
    Pi_d = jnp.linalg.inv(Sigma_d)
    Mu_a = mu_a*jnp.ones(W.shape[1])
    Sigma = Sigma_a - Sigma_a @ W.T @ jnp.linalg.inv(Sigma_d + W @ Sigma_a @ W.T) @ W @ Sigma_a
    mu = Mu_a + Sigma_a @ W.T @ QR(jnp.identity(len(d)) + Pi_d @ W @ Sigma_a @ W.T, Pi_d @ (d - W @ Mu_a))
    return mu, Sigma

def posterior_map_moments_from_samples(samples, d, nside, n_phase, n_wav, wl, line_profile, dmat, use_indices=None):
    """
    Calculate the posterior mean and variance maps by averaging over MCMC samples.
    
    Parameters
    ----------
    samples : dict
        MCMC samples containing arrays for each parameter.
    d : array-like
        Observed data vector.
    nside : int
        HEALPix resolution parameter.
    n_phase : int
        Number of observational phases.
    n_wav : int
        Number of wavelength points.
    wl : array-like
        Wavelength grid of the observations.
    line_profile : array-like
        Intrinsic local line profile (template).
    dmat : array-like
        Precomputed angular distance matrix for the HEALPix grid.
    use_indices : array-like, optional
        Indices of MCMC samples to use for the calculation. If None, use all samples"""

    # Extract parameters from samples
    v_all       = jnp.asarray(samples['v'])
    cosi_all    = jnp.asarray(samples['cosi'])
    i_all       = jnp.arccos(cosi_all)
    u_all       = jnp.asarray(samples['u'])
    log_w_all   = jnp.asarray(samples['log_w'])   # shape (S, n_phase)
    sigma_d_all = jnp.asarray(samples['sigma_d'])
    mu_a_all    = jnp.asarray(samples['mu_a'])
    sigma_a_all = jnp.asarray(samples['sigma_a'])
    log_ell_all = jnp.asarray(samples['log_ell'])
    ell_all     = jnp.exp(log_ell_all)

    S_total = v_all.shape[0]

    # Determine which samples to use
    if use_indices is None:
        idx_array = jnp.arange(S_total)
    else:
        idx_array = jnp.asarray(use_indices)
    S = idx_array.shape[0]

    # Initialize accumulators for mean and variance calculations
    mu_list = []
    npix = hp.nside2npix(nside)
    Sigma_diag_sum = jnp.zeros(npix)

    # Loop over selected MCMC samples
    for idx in np.array(idx_array):
        v       = v_all[idx]
        i       = i_all[idx]
        u       = u_all[idx]
        w       = jnp.exp(log_w_all[idx])
        sigma_d = sigma_d_all[idx]
        mu_a    = mu_a_all[idx]
        sigma_a = sigma_a_all[idx]
        ell     = ell_all[idx]

        mu_s, Sigma_s = mu2(w, v, i, u, mu_a, sigma_a, sigma_d, ell, d, nside, n_phase, n_wav, wl, line_profile, dmat)  # (npix,), (npix,npix)

        mu_list.append(mu_s)
        Sigma_diag_sum = Sigma_diag_sum + jnp.diag(Sigma_s)

    mu_stack = jnp.stack(mu_list, axis=0)

    # posterior mean map
    mu_bar = jnp.mean(mu_stack, axis=0)

    # posterior variance map using the law of total variance
    diff2_mean = jnp.mean((mu_stack - mu_bar[None, :]) ** 2, axis=0)
    var_diag = Sigma_diag_sum / S + diff2_mean

    return mu_bar, var_diag

def main():
    # ====== generate synthetic data ======
    wl0 = 656.28
    n_wav = 100
    wl = jnp.linspace(wl0 - 0.15, wl0 + 0.15, n_wav)
    nu0 = 1e7 / wl0
    nu = 1e7 / wl
    line_profile = 1 - 0.8 * norm.pdf(nu, nu0, 0.3)
    n_phase = 8
    nside = 8

    true_vsini = 10
    true_i = jnp.deg2rad(jnp.array(range(10,81,10)))
    true_v = true_vsini/jnp.sin(true_i)
    true_u = 0.5
    true_w = jnp.array([1.00, 0.98, 1.03, 0.99, 1.01, 0.97, 1.02, 1.00])

    pi = np.pi
    npix = healpy.nside2npix(nside)
    theta, phi = healpy.pix2ang(nside, np.arange(npix))
    def add_s(m, th, ph, r, i):
        mask = healpy.rotator.angdist([th, ph], [theta, phi]) < r
        m[mask] *= i
        return m
    map1 = add_s(np.ones(npix), pi/4, 0, pi/6, 0.1)
    map2 = add_s(np.ones(npix), pi/3, pi/4, pi/6, 0.1)
    map2 = add_s(map2, 2*pi/3, -2*pi/3, pi/5, 0.1)
    map3 = add_s(map2.copy(), pi/6, -3*pi/4, pi/8, 0.1)
    map3 = add_s(map3, 7*pi/12, 3*pi/4, pi/8, 0.1)
    true_map = map3

    d = jnp.zeros((len(true_i), n_phase*len(wl)))
    for m in range(len(true_v)):
        d = d.at[m].set(generate(true_map, wl, line_profile, true_v[m], true_i[m], true_u, true_w, n_phase, nside))

    # ====== run MCMC estimation ======
    output_path = Path("./results_map3")
    output_path.mkdir(exist_ok=True)
    npix = hp.nside2npix(nside)
    vmap_func = jax.vmap(partial(hp.pix2ang, 'ring', nside))
    th, ph = vmap_func(jnp.arange(npix))
    dmat = angular_distance(th[:, None], ph[:, None], th[None, :], ph[None, :])
    
    for m in range(len(true_v)):
        samples = run_mcmc_estimation(d[m], wl, line_profile, dmat, n_phase, nside, n_wav, key_seed=0)
        results_file = output_path / f"mcmc_samples_i={jnp.rad2deg(true_i[m]):.2f}.npz"
        save_data = {k: np.asarray(v) for k, v in samples.items()}
        np.savez(results_file, **save_data)

    # ====== draw mean and uncertainty maps ======
    for m in range(len(true_v)):
        samples_file = output_path / f"mcmc_samples_i={jnp.rad2deg(true_i[m]):.2f}.npz"
        samples = np.load(samples_file)
        mu_bar, var_diag = posterior_map_moments_from_samples(samples, d[m], nside, n_phase, n_wav, wl, line_profile, dmat)
        healpy.mollview(mu_bar, title=f"Posterior mean map (i={jnp.rad2deg(true_i[m]):.2f} deg)", cmap="inferno", unit="Intensity", flip='geo')
        plt.savefig(output_path / f"posterior_mean_map_i={jnp.rad2deg(true_i[m]):.2f}.png")
        plt.clf()
        healpy.mollview(jnp.sqrt(var_diag), title=f"Posterior std dev map (i={jnp.rad2deg(true_i[m]):.2f} deg)", cmap="viridis", unit="Std Dev", flip='geo')
        plt.savefig(output_path / f"posterior_std_map_i={jnp.rad2deg(true_i[m]):.2f}.png")
        plt.clf()

if __name__ == "__main__":
    main()
