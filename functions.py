import numpy as np
import astropy.units as u
from speclite import filters
from astropy.constants import h, c, k_B, sigma_sb
from astropy.cosmology import Planck18
from lsstetc import s0

def F_blackbody(wavelength, T = 5000 * u.K):
    """
    Compute the blackbody surface flux density (per unit wavelength)
    in the rest frame of the blackbody.

    Args
    ----------
    wavelength : Quantity
        Wavelength(s), must have units (e.g. u.AA, u.nm, u.um).
    T (temperature) : Quantity
        Temperature with units of Kelvin.

    Returns
    -------
    flux_density : array
        Flux density (erg / s / cm² / Å), i.e., total emitted flux from the surface.
    """

    exponent = (h.cgs * c.cgs) / (wavelength.to(u.cm) * k_B.cgs * T.to(u.K))
    # specific intensity from a blackbody emitter
    B = (2 * h.cgs * c.cgs**2) / (wavelength.to(u.cm)**5 * (np.exp(exponent.value) - 1))
    # surface flux density (integrated over solid angle)
    F = np.pi * B

    return F.to(u.erg / (u.s * u.cm**2 * u.AA))

def L_blackbody(wavelength, T = 5000 * u.K):
    """
    Compute the blackbody rest-frame luminosity at a given wavelength and temperature.

    Args
    wavelength : array
        Wavelength(s), must have units (e.g. u.AA, u.nm, u.um).
    T (temperature) : float
        Temperature with units of Kelvin.

    Returns
    -------
    luminosity : array
        Luminosity (erg / s), i.e., total emitted flux from the surface.
    """
    L_sun = 3.826e33 * u.erg / u.s # Solar luminosity
    L_bol = 1e9 * L_sun # a standard Bolometric luminosity of a galaxy in Solar luminosities
    # Calculate the effective radius of the blackbody in cm
    R_eff = np.sqrt(L_bol / (4 * np.pi * sigma_sb.to(u.erg / (u.s * u.cm**2 * u.K**4)) * T**4))
    return 4 * np.pi * R_eff**2 * F_blackbody(wavelength, T)

def F_observed(z, wavelength, T = 5000 * u.K):
    """
    Compute the observed flux in wavelength of a blackbody at a given wavelength and temperature.

    Args
    z : float
        Redshift.
    wavelength : array
        Wavelength(s), must have units (e.g. u.AA, u.nm, u.um).
    T (temperature) : float
        Temperature with units of Kelvin.

    Returns
    -------
    flux : array
        Flux in wavelength (erg / s / cm² / Å).
    """
    if not isinstance(z, np.ndarray):
        z = np.array([z])
    z = z.reshape(-1, 1)
    return L_blackbody(wavelength.reshape(1, -1) / (1+z), T) / ((1+z) * (4 * np.pi * Planck18.luminosity_distance(z).to(u.cm)**2))

def F_b(z, f, T = 5000 * u.K):
    """
    Compute the integrated band (filter) photon flux of a blackbody at a given redshift, filter, temperature, and radius.

    Args
    z : float (or array)
        Redshift.
    f (filter) : str
        LSST filter name. "u", "g", "r", "i", "z", "y"

    Returns
    -------
    photon flux : float (or array)
        Flux in band (1 / s / cm²).
    """
    loaded_filter = filters.load_filter("lsst2023-"+f)
    wlen = loaded_filter.wavelength * u.AA
    return np.trapezoid(F_observed(z, wlen, T) * loaded_filter(wlen) * (wlen / (h * c).to(u.erg * u.Angstrom)), wlen)

def photon_count(t_exp, z, f, A = 33.2 * u.m**2, T = 5000 * u.K):
    """
    Compute the number of photons in a given exposure time, redshift, filter, and temperature.

    Args
    t_exp : float
        Exposure time in seconds.
    z (redshift) : float
        Redshift.
    f (filter) : str
        LSST filter name. "u", "g", "r", "i", "z", "y"
    T (temperature) : float
        Temperature with units of Kelvin.

    Returns
    -------
    photons : float
        Number of photons in the band.
    """
    # convert t_exp to seconds if it's a float, otherwise ensure it has time units
    if isinstance(t_exp, (int, float)):
        t_exp = t_exp * u.s
    else:
        t_exp = t_exp.to(u.s)
    return F_b(z, f, T) * t_exp * A.to(u.cm**2)

def M_b(z, f, T = 5000 * u.K):
    """
    Compute the photo flux magnitude of a blackbody at a given redshift, filter, temperature, and radius.
    Uses photon flux and is calibrated to match the s0 zeropoint system for LSST.

    Args
    z : float (or array)
        Redshift.
    f (filter) : str
        LSST filter name. "u", "g", "r", "i", "z", "y"
    T (temperature) : float
        Temperature with units of Kelvin.

    Returns
    -------
    magnitude : float (or array)
        Magnitude in the input filter band based on photon flux.
    """
    # convert z to array if it is a scalar
    if not isinstance(z, np.ndarray):
        z = np.array([z])
    
    # Get photon flux directly from F_b (photons/s/cm²)
    photon_flux = F_b(z, f, T)
    

    # s0 is in photons/s, so we need to account for the effective area
    A = 319/9.6 # LSST effective area in m^2 from lsstetc.py
    A_cm2 = A * 1e4 * u.cm**2 # convert to cm^2
    
    # Get the photons/sec for the whole detector
    photon_flux_pixel = photon_flux * A_cm2
    
    # Convert to magnitude using the s0 zeropoint system
    # where mag=24.0 corresponds to s0 photons/s/pixel
    magnitude = 24.0 - 2.5 * np.log10(photon_flux_pixel / (s0[f] * (1 / (u.s))))
    
    return magnitude