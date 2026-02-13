import jax.numpy as jnp
import jax
from functools import partial
import healjax as hp
from const import c0
    
def incline(theta0, phi0, alpha):
    """
    Apply inclination rotation to spherical coordinates.
    
    Parameters
    ----------
    theta0, phi0 : jnp.ndarray
        Original spherical coordinates.
    alpha : float
        Rotation angle in radians (pi/2 - inclination).
    """
    # Convert spherical to Cartesian
    x0 = jnp.sin(theta0) * jnp.cos(phi0)
    y0 = jnp.sin(theta0) * jnp.sin(phi0)
    z0 = jnp.cos(theta0)

    # Apply rotation matrix around the y-axis
    x = jnp.cos(alpha) * x0 + jnp.sin(alpha) * z0
    y = y0
    z = -jnp.sin(alpha) * x0 + jnp.cos(alpha) * z0

    # Convert back to spherical
    theta = jnp.arccos(z)
    phi = jnp.arctan2(y, x)

    return theta, phi

def limb_darkening(u, mu):
    """Linear limb darkening law."""
    return 1 - u * (1 - mu) #

def limb_darkening2(u_1, u_2, mu):
    """Quadratic limb darkening law."""
    return 1 - u_1 * (1 - mu) - u_2 * (1 - mu)**2 #

def doppler_shift(vlos):
    """
    Calculate Doppler factor (1 + beta) / sqrt(1 - beta^2).
    
    Parameters
    ----------
    vlos : jnp.ndarray
        Line-of-sight velocity in km/s.
    """
    c_km_s = c0 * 1e-3  # Convert m/s to km/s
    beta = vlos / c_km_s
    return (1 + beta) / jnp.sqrt(1 - beta**2) #

def angular_distance(th1, ph1, th2, ph2):
    """
    Calculate the great-circle distance between two points on a sphere.
    Used for Gaussian Process kernel construction.
    """
    cosg = jnp.cos(th1) * jnp.cos(th2) + jnp.sin(th1) * jnp.sin(th2) * jnp.cos(ph1 - ph2)
    cosg = jnp.clip(cosg, -1.0, 1.0) #
    return jnp.arccos(cosg) #

def get_exposure_matrix(nside, vrot, inclination, u1, u2, phase, wavelengths, line_profile):
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
    u1, u2 : float
        Limb darkening coefficients.
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
    ld = limb_darkening2(u1, u2, mu) #

    visible = ((-jnp.pi/2 < phi) & (phi < jnp.pi/2)).astype(float) #
    weight = visible * jnp.sin(theta) * jnp.cos(phi) #

    # Design matrix M for this specific phase
    M = weight[None, :] * ld[None, :] * interped #
    return M

def get_full_design_matrix(nside, vrot, inclination, u1, u2, obs_times, P, wavelengths, line_profile, w):
    """
    Compute the full concatenated design matrix W for all observation phases.
    """
    phases = obs_times / P #

    def scaled_matrix(phase, wk):
        M = get_exposure_matrix(nside, vrot, inclination, u1, u2, phase, wavelengths, line_profile)
        return wk * M # Apply scaling factor w_k

    # Vectorize across phases and scale factors
    M_stack = jax.vmap(scaled_matrix)(phases, w) #
    
    # Reshape to (N_phase * N_wav, N_pix)
    n_phase = len(obs_times)
    W = M_stack.reshape((n_phase * len(wavelengths), -1)) #
    return W