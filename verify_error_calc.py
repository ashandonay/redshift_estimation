#!/usr/bin/env python3
"""
Simple verification script to compare the fast method vs traditional GalSim withFlux.
This ensures the fast approach produces identical results to the standard method.
"""

import numpy as np
import galsim
import time
import matplotlib.pyplot as plt
from lsstetc import ETC

def main(test_mags=np.linspace(30.0, 20.0, 500), nvisits=np.array([20])):
    print("Verifying the fast method matches the traditional GalSim withFlux")
    print("=" * 70)
    
    # Create a simple galaxy profile
    psf = galsim.Gaussian(fwhm=0.67)
    gal = galsim.Sersic(n=1.0, half_light_radius=0.2)
    profile = galsim.Convolve(psf, gal)

    # Set up ETC instance
    etc = ETC('i', profile=profile, pixel_scale=0.2, stamp_size=31, threshold=0.0)

    print(f"Profile: {type(profile).__name__}")
    print(f"Stamp size: {etc.stamp_size}x{etc.stamp_size}")
    print(f"Pixel scale: {etc.pixel_scale} arcsec")
    print()
    
    # Method 1: Traditional GalSim withFlux approach
    print("Method 1: Traditional GalSim withFlux + drawImage")
    print("-" * 50)
    start_time = time.time()
    
    galsim_imgs = []
    snrs = []
    mag_errs = []
    for m in test_mags:
        flux = etc.s0 * 10**(-0.4*(m - 24.0)) * nvisits * etc.visit_time
        profile_with_flux = profile.withFlux(flux)
        img = galsim.ImageD(etc.stamp_size, etc.stamp_size, scale=etc.pixel_scale)
        profile_with_flux.drawImage(image=img)
        galsim_imgs.append(img.array)
        snr, signal, noise = etc.SNR(profile, m, nvisits)
        snrs.append(snr)
        mag_errs.append(etc.err(profile, m, nvisits))

    galsim_imgs = np.array(galsim_imgs)
    snrs = np.array(snrs)
    mag_errs = np.array(mag_errs)
    traditional_time = time.time() - start_time
    
    print(f"Galsim image shape: {galsim_imgs.shape}")

    # Method 2: Ultra-fast approach
    print("\nMethod 2: Fast approach")
    print("-" * 50)
    start_time = time.time()
    fast_imgs, _ = etc.get_pixel_values(test_mags, nvisits)
    fast_snrs = etc.batch_snr(test_mags, nvisits)
    fast_mag_errs = etc.mag_err(test_mags, nvisits)

    fast_time = time.time() - start_time
    
    print(f"Fast image shape: {fast_imgs.shape}")
    
    # Comparison
    print("\nComparison Results")
    print("=" * 70)
    
    # Check if arrays are identical (with slightly relaxed tolerance for floating-point precision)
    are_identical = np.allclose(galsim_imgs, fast_imgs, rtol=1e-8, atol=1e-10)
    print(f"Pixel values are identical: {are_identical}")

    if not are_identical:
        print("FAILURE: Pixel values don't match!")
        print(f"Pixel values differences: {np.abs(galsim_imgs - fast_imgs)}")
        # Show differences
        diff = np.abs(galsim_imgs - fast_imgs)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"Maximum absolute difference: {max_diff:.2e}")
        print(f"Mean absolute difference: {mean_diff:.2e}")
        
        # Find where differences occur
        diff_positions = np.where(diff > 1e-12)
        if len(diff_positions[0]) > 0:
            print(f"Differences found at {len(diff_positions[0])} pixel positions")
            print("First few differences:")
            for i in range(min(5, len(diff_positions[0]))):
                mag_idx, y, x = diff_positions[0][i], diff_positions[1][i], diff_positions[2][i]
                print(f"  Mag {mag_idx}, Pixel ({y},{x}): Traditional={galsim_imgs[mag_idx,y,x]:.6e}, Ultra-fast={fast_imgs[mag_idx,y,x]:.6e}")

    snr_match = np.allclose(snrs, fast_snrs, rtol=1e-10)
    print(f"SNR values are identical: {snr_match}")
    
    if not snr_match:
        print("FAILURE: SNR values don't match!")
        print(f"SNR differences: {snrs[np.where(~snr_match)]}")

    mag_err_match = np.allclose(mag_errs, fast_mag_errs, rtol=1e-10)
    print(f"Magnitude error values are identical: {mag_err_match}")
    
    if not mag_err_match:
        print("FAILURE: Magnitude error values don't match!")
        print(f"Magnitude error differences: {mag_errs[np.where(~mag_err_match)]}")
    
    
    # Performance comparison
    print("\nPerformance Comparison")
    print("-" * 50)
    
    print(f"Traditional method: {traditional_time:.4f}s")
    print(f"Ultra-fast method: {fast_time:.4f}s")
    print(f"Speedup: {traditional_time/fast_time:.1f}x")
    
    
    print("\n" + "=" * 70)
    if are_identical and snr_match:
        print("ALL VERIFICATIONS PASSED! Ultra-fast method is working correctly.")
    else:
        print("Some verifications failed. Check the differences above.")

if __name__ == "__main__":
    main()
