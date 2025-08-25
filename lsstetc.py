
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
    def __init__(self, band, profile=None, pixel_scale=None, stamp_size=None, threshold=0.0,
                 nvisits=None, visit_time=30.0):
        
        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.band = band
        if nvisits is None:
            nvisits = fiducial_nvisits[band]
        self.nvisits = nvisits
        self.visit_time =  visit_time
        self.exptime = self.nvisits * self.visit_time
        self.sky = sbar[band] * self.exptime * self.pixel_scale**2
        self.sigma_sky = np.sqrt(self.sky)
        self.s0 = s0[band]
        self._base_img = self._calculate_base_profile(profile)
                
    def draw(self, profile, mag, noise=False):
        img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        flux = self.s0 * 10**(-0.4*(mag - 24.0)) * self.exptime
        profile = profile.withFlux(flux)
        profile.drawImage(image=img)
        if noise:
            gd = galsim.GaussianNoise(bd, sigma=self.sigma_sky)
            img.addNoise(gd)
        return img

    def SNR(self, profile, mag):
        img = self.draw(profile, mag, noise=False)
        mask = img.array > (self.threshold * self.sigma_sky)
        imgsqr = img.array**2*mask
        signal = imgsqr.sum()
        noise = np.sqrt((imgsqr * self.sky).sum())
        return signal / noise, signal, noise

    def nphot(self, mag):
        return self.s0 * 10**(-0.4*(mag - 24.0)) * self.exptime

    def err(self, profile, mag):
        snr, signal, noise = self.SNR(profile, mag)
        return 2.5 / np.log(10) / snr

    def display(self, profile, mag, noise=True):
        img = self.draw(profile, mag, noise)
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
    
    def get_pixel_values(self, mags):
        """Ultra-fast method using pre-calibrated base profile for magnitude calculations.
        
        This method uses pre-computed base profile values to rapidly scale
        for different magnitudes without redrawing images or recalculating xValue.
        
        Parameters
        ----------
        profile : galsim.GSObject
            Galaxy profile to use for calculations
        mags : array-like
            Array of magnitude values
            
        Returns
        -------
        tuple
            (pixels_array, mask_array) computed using pre-calibrated base profile
        """
        
        mags = np.asarray(mags)
        n_mags = len(mags)
        
        # Initialize arrays
        pixels_array = np.zeros((n_mags, self.stamp_size, self.stamp_size))
        mask_array = np.zeros((n_mags, self.stamp_size, self.stamp_size), dtype=bool)
        
        # Vectorized flux calculation for all magnitudes at once
        fluxes = self.s0 * 10**(-0.4*(mags - 24.0)) * self.exptime
        
        # Single NumPy operation: scale base profile for all magnitudes
        pixels_array = fluxes[:, np.newaxis, np.newaxis] * self._base_img
        
        # Vectorized mask calculation
        mask_array = pixels_array > (self.threshold * self.sigma_sky)
        
        return pixels_array, mask_array

    def batch_snr(self, mags):
        """
        Calculates the flux SNR for multiple magnitudes efficiently.
        
        This method computes both pixel weights and SNR using the ultra-fast
        vectorized approach, avoiding the need to draw images for each magnitude.
        It uses the pre-calibrated base profile to scale the profile for each magnitude.
        
        Parameters
        ----------
        mags : array-like
            Array of magnitude values
            
        Returns
        -------
        tuple
            snr_array: 1D array of SNR values for each magnitude
        """
        # Get pixel values using ultra-fast method
        pixels_array, mask_array = self.get_pixel_values(mags)
        
        snr_array = self._compute_snr_from_pixels(pixels_array, mask_array)
        
        return snr_array

    def _compute_snr_from_pixels(self, pixels_array, mask_array):
        """Compute SNR directly from pixel weights (no image drawing needed)."""
        # Handle both 2D and 3D arrays
        if pixels_array.ndim == 3:
            # 3D array: (n_mags, height, width)
            n_mags = pixels_array.shape[0]
            snr_array = np.zeros(n_mags)
            
            for i in range(n_mags):
                # Apply threshold mask to get valid pixels for this magnitude
                valid_pixels = pixels_array[i][mask_array[i]]
                
                if len(valid_pixels) == 0:
                    snr_array[i] = 0.0
                    continue
                
                # Compute weighted SNR using the pixel weights
                signal = (valid_pixels**2).sum()
                noise = np.sqrt((valid_pixels**2 * self.sky).sum())
                
                if noise == 0:
                    snr_array[i] = float('inf') if signal > 0 else 0.0
                else:
                    snr_array[i] = signal / noise
            
            return snr_array
        else:
            # 2D array: single magnitude
            # Apply threshold mask to get valid pixels
            valid_pixels = pixels_array[mask_array]
            
            if len(valid_pixels) == 0:
                return 0.0
            
            # Compute weighted SNR using the pixel weights
            signal = (valid_pixels**2).sum()
            noise = np.sqrt((valid_pixels**2 * self.sky).sum())
            
            if noise == 0:
                return float('inf') if signal > 0 else 0.0
            
            return signal / noise

    def mag_err(self, mags):
        """Compute magnitude errors from pixel weights using ultra-fast method.
        
        This method computes magnitude errors efficiently using the ultra-fast
        pixel weight computation and SNR calculation.
        
        Parameters
        ----------
        profile : galsim.GSObject
            Galaxy profile to use for calculations
        mags : array-like
            Array of magnitude values
            
        Returns
        -------
        numpy.ndarray
            Array of magnitude errors corresponding to each magnitude
        """
        # Get SNR values using ultra-fast method
        snr_array = self.batch_snr(mags)
        
        # Calculate magnitude errors
        errs = 2.5 / (np.log(10) * snr_array)
        
        return errs