
# Adapted from: https://github.com/jmeyers314/LSST_ETC/blob/master/lsstetc.py

"""An exposure time calculator for LSST.  Uses GalSim to draw a galaxy with specified magnitude,
shape, etc, and then uses the same image as the optimal weight function.  Derived from D. Kirkby's
notes on deblending.
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import galsim

# Some constants
# --------------
#
# Photometric zeropoints from SMTN-002 (v1.9 throughputs)
# These are AB magnitudes that produce 1 count per second
# Based on syseng_throughputs v1.9 with triple silver mirror coatings
# and as-measured filter/lens/detector throughputs
# https://smtn-002.lsst.io
# Encodes the physical properties of the telescope and detector system including effective area and throughput.
s0 = {'u': 26.52,
      'g': 28.51,
      'r': 28.36,
      'i': 28.17,
      'z': 27.78,
      'y': 26.82}

# Sky brightnesses in AB mag / arcsec^2 (zenith, dark sky)
# From SMTN-002: https://smtn-002.lsst.io
# Based on dark sky spectrum from UVES/Gemini/ESO, normalized to match SDSS observations
B = {'u': 23.05,
     'g': 22.25,
     'r': 21.2,
     'i': 20.46,
     'z': 19.61,
     'y': 18.6}

# Sky brightness per arcsec^2 per second
# At sky magnitude B[k]: flux = 10^(-0.4*(B[k] - s0[k])) photons/sec/arcsec^2
sbar = {}
for k in B:
    sbar[k] = 10**(-0.4*(B[k] - s0[k]))

# Number of visits over 10 years from LSST Science Book (table 1.1)
fiducial_nvisits = {'u': 70,
                    'g': 100,
                    'r': 230,
                    'i': 230,
                    'z': 200,
                    'y': 200}

# And some random numbers for drawing
bd = galsim.BaseDeviate(1)


class ETC(object):
    def __init__(self, band, profile=None, pixel_scale=0.2, stamp_size=31, threshold=0.0, 
                 exposure_time=15.0, n_exp_per_visit=2, read_noise=8.8, dark_current=0.2):
        
        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.band = band
        self.exposure_time = exposure_time  # seconds per exposure (15s for LSST)
        self.n_exp_per_visit = n_exp_per_visit  # number of exposures per visit (2 for LSST)
        self.visit_time = exposure_time * n_exp_per_visit  # total time per visit (30s for LSST)
        self.read_noise = read_noise  # electrons per pixel per exposure
        self.dark_current = dark_current  # electrons per pixel per second
        self.s0 = s0[band]
        self._base_img = self._calculate_base_profile(profile)
                
    def draw(self, profile, mag, nvisits, noise=False):
        img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        # At zeropoint magnitude: 1 photon/sec
        # At magnitude m: 10^(-0.4*(m - zeropoint)) photons/sec
        flux = 10**(-0.4*(mag - self.s0)) * nvisits * self.visit_time
        profile = profile.withFlux(flux)
        profile.drawImage(image=img)
        sigma_sky = np.sqrt(self.get_sky_var(nvisits))
        if noise:
            gd = galsim.GaussianNoise(bd, sigma=sigma_sky)
            img.addNoise(gd)
        return img
    
    def SNR(self, profile, mag, nvisits):
        img = self.draw(profile, mag, nvisits, noise=False)
        sigma_sky = np.sqrt(self.get_sky_var(nvisits))
        mask = img.array > (self.threshold * sigma_sky)
        imgsqr = img.array**2*mask
        signal = imgsqr.sum()
        # Total variance: sky + source Poisson + dark current + read noise
        # All variances in electrons^2
        sky_var = self.get_sky_var(nvisits)
        dark_var = self.get_dark_var(nvisits)
        source_var = img.array * mask
        read_var = self.get_read_var(nvisits)
        total_var = sky_var + source_var + dark_var + read_var
        noise = np.sqrt((imgsqr * total_var).sum())
        return signal / noise, signal, noise

    def nphot(self, mag, nvisits):
        return 10**(-0.4*(mag - self.s0)) * nvisits * self.visit_time

    def err(self, profile, mag, nvisits):
        snr, signal, noise = self.SNR(profile, mag, nvisits)
        return 2.5 / np.log(10) / snr

    def display(self, profile, mag, nvisits, noise=True):
        img = self.draw(profile, mag, nvisits, noise)
        plt.imshow(img.array, cmap=cm.Greens)
        plt.colorbar()
        plt.show()

    def _calculate_base_profile(self, profile):
        """
        Creates a base profile image once for fast magnitude error calculations.
        
        This method draws the base profile with unit flux once, then
        scales it for different magnitudes using simple NumPy operations.
        
        Parameters
        ----------
        profile : galsim.GSObject
            Galaxy profile to calibrate
        """
        
        # Create a unit flux profile and draw it once
        unit_profile = profile.withFlux(1.0)
        base_img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        unit_profile.drawImage(image=base_img)
        
        return base_img.array.copy()

    def get_sky_var(self, nvisits):
        """Get the sky variance in photons per pixel."""
        exptime = np.atleast_1d(nvisits) * self.visit_time
        sky = sbar[self.band] * exptime * self.pixel_scale**2
        return sky
    
    def get_read_var(self, nvisits):
        """Get the read noise variance per pixel (in electrons^2).
        
        Each visit has n_exp_per_visit exposures, each with independent read noise.
        For LSST: 2 exposures per visit, so read_var = read_noise^2 * 2 * nvisits
        """
        nvisits = np.atleast_1d(nvisits)
        n_exposures = nvisits * self.n_exp_per_visit
        return self.read_noise**2 * n_exposures
    
    def get_dark_var(self, nvisits):
        """Get the dark current variance per pixel (in electrons).
        
        Dark current follows Poisson statistics, so variance = mean.
        Mean dark current = dark_current_rate * total_exposure_time
        For LSST: 2 exposures of 15s each = 30s total per visit
        """
        nvisits = np.atleast_1d(nvisits)
        total_exptime = nvisits * self.visit_time
        return self.dark_current * total_exptime
    
    def get_pixel_values(self, mags, nvisits):
        """
        Vectorized method using pre-calibrated base profile for magnitude calculations.
        
        This method uses pre-computed base profile values to rapidly scale
        for different magnitudes without redrawing images or recalculating xValue.
        
        Parameters
        ----------
        mags : array-like
            Array of magnitude values
        nvisits : array-like
            Array of number of visits
            
        Returns
        -------
        tuple
            pixels_array: Array of image pixel values
            mask_array: Array of mask values
        """
        
        mags = np.atleast_1d(mags)
        nvisits = np.atleast_1d(nvisits)
        
        if nvisits.shape[-1] != 1:
            nvisits_reshaped = nvisits[..., np.newaxis]
        else:
            nvisits_reshaped = nvisits
        
        # Vectorized flux calculation for all magnitudes at once
        # At zeropoint magnitude: 1 photon/sec; at mag m: 10^(-0.4*(m - zeropoint))
        fluxes = 10**(-0.4*(mags - self.s0)) * nvisits_reshaped * self.visit_time
        
        # Single NumPy operation: scale base profile for all magnitudes
        pixels_array = fluxes[..., np.newaxis, np.newaxis] * self._base_img

        sigma_sky = np.sqrt(self.get_sky_var(nvisits_reshaped))
        # Vectorized mask calculation
        mask_array = pixels_array > (self.threshold * sigma_sky[..., np.newaxis, np.newaxis])
        
        return pixels_array, mask_array

    def batch_snr(self, mags, nvisits):
        """
        Calculates the flux SNR for multiple magnitudes efficiently.
        
        This method computes both pixel weights and SNR using the ultra-fast
        vectorized approach, avoiding the need to draw images for each magnitude.
        It uses the pre-calibrated base profile to scale the profile for each magnitude.
        
        Parameters
        ----------
        mags : array-like
            Array of magnitude values
        nvisits : array-like
            Array of number of visits
            
        Returns
        -------
        tuple
            snr_array: 1D array of SNR values for each magnitude
        """
        # Get pixel values using ultra-fast method
        pixels_array, mask_array = self.get_pixel_values(mags, nvisits)
        
        snr_array = self._compute_snr_from_pixels(nvisits, pixels_array, mask_array)
        
        return snr_array

    def _compute_snr_from_pixels(self, nvisits, pixels_array, mask_array):
        """
        Compute the flux SNR directly from pixel weights (no image drawing needed).
        
        Parameters
        ----------
        nvisits : array-like
            Array of number of visits
        pixels_array : numpy.ndarray
            Array of pixel values
        mask_array : numpy.ndarray
            Array of mask values

        Returns
        -------
        numpy.ndarray
            Array of flux SNR values
        """
        nvisits = np.atleast_1d(nvisits)
        if nvisits.shape[-1] != 1:
            nvisits_reshaped = nvisits[..., np.newaxis]
        else:
            nvisits_reshaped = nvisits
        
        masked_pixels = pixels_array * mask_array
        signal = (masked_pixels**2).sum(axis=(-2, -1))
        
        # Total variance: sky + source Poisson + dark current + read noise
        # All variances in electrons^2 (assuming gain=1)
        sky_var = self.get_sky_var(nvisits_reshaped)[..., np.newaxis, np.newaxis]
        src_var = masked_pixels  # Source Poisson noise (mean=variance)
        dark_var = self.get_dark_var(nvisits_reshaped)[..., np.newaxis, np.newaxis]
        read_var = self.get_read_var(nvisits_reshaped)[..., np.newaxis, np.newaxis]
        total_var = sky_var + src_var + dark_var + read_var
        noise = np.sqrt((masked_pixels**2 * total_var).sum(axis=(-2, -1)))
        
        snr_array = np.where(noise == 0, 
                            np.where(signal > 0, float('inf'), 0.0), 
                            signal / noise)
        return snr_array

    def mag_err(self, mags, nvisits):
        """
        Compute magnitude errors from pixel weights using ultra-fast method.
        
        This method computes magnitude errors efficiently using the ultra-fast
        pixel weight computation and SNR calculation.
        
        Parameters
        ----------
        mags : array-like
            Array of magnitude values
        nvisits : array-like
            Array of number of visits
            
        Returns
        -------
        numpy.ndarray
            Array of magnitude errors corresponding to each magnitude
        """
        # Get SNR values using vectorized method
        snr_array = self.batch_snr(mags, nvisits)

        # Calculate magnitude errors
        errs = 2.5 / (np.log(10) * snr_array)
        
        return errs