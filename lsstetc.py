
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
# LSST effective area in meters^2
A = 319/9.6  # etendue / FoV.  I *think* this includes vignetting

# zeropoints from DK notes in photons per second per pixel
# should eventually compute these on the fly from filter throughput functions.
s0 = {'u': A*0.732,
      'g': A*2.124,
      'r': A*1.681,
      'i': A*1.249,
      'z': A*0.862,
      'y': A*0.452}
# Sky brightnesses in AB mag / arcsec^2.
# stole these from http://www.lsst.org/files/docs/gee_137.28.pdf
# should eventually construct a sky SED (varies with the moon phase) and integrate to get these
B = {'u': 22.8,
     'g': 22.2,
     'r': 21.3,
     'i': 20.3,
     'z': 19.1,
     'y': 18.1}
# number of visits
# From LSST Science Book
fiducial_nvisits = {'u': 56,
                    'g': 80,
                    'r': 180,
                    'i': 180,
                    'z': 164,
                    'y': 164}
# Sky brightness per arcsec^2 per second
sbar = {}
for k in B:
    sbar[k] = s0[k] * 10**(-0.4*(B[k]-24.0))

# And some random numbers for drawing
bd = galsim.BaseDeviate(1)


class ETC(object):
    def __init__(self, band, profile=None, pixel_scale=None, stamp_size=None, threshold=0.0, visit_time=30.0):
        
        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.band = band
        self.visit_time =  visit_time
        self.s0 = s0[band]
        self._base_img = self._calculate_base_profile(profile)
                
    def draw(self, profile, mag, nvisits, noise=False):
        img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        flux = self.s0 * 10**(-0.4*(mag - 24.0)) * nvisits * self.visit_time
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
        noise = np.sqrt((imgsqr * self.get_sky_var(nvisits)).sum())
        return signal / noise, signal, noise

    def nphot(self, mag, nvisits):
        return self.s0 * 10**(-0.4*(mag - 24.0)) * nvisits * self.visit_time

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
        n_mags = len(mags)
        n_exps = len(nvisits)
        
        # Initialize arrays
        pixels_array = np.zeros((n_exps, n_mags, self.stamp_size, self.stamp_size))
        mask_array = np.zeros((n_exps, n_mags, self.stamp_size, self.stamp_size), dtype=bool)
        
        # Vectorized flux calculation for all magnitudes at once
        fluxes = self.s0 * 10**(-0.4*(mags.reshape(1, -1) - 24.0)) * nvisits.reshape(-1, 1) * self.visit_time
        
        # Single NumPy operation: scale base profile for all magnitudes
        pixels_array = fluxes[..., np.newaxis, np.newaxis] * self._base_img

        sigma_sky = np.sqrt(self.get_sky_var(nvisits))
        # Vectorized mask calculation
        mask_array = pixels_array > (self.threshold * sigma_sky)[:, np.newaxis, np.newaxis, np.newaxis]
        
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
        masked_pixels = pixels_array * mask_array
        # Compute SNR for each batch element
        signal = (masked_pixels**2).sum(axis=(-2, -1))  # Sum over height, width
        noise = np.sqrt((masked_pixels**2 * self.get_sky_var(nvisits)[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=(-2, -1)))
        
        # Handle division by zero
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