#pragma once
#include <data_types/fourierseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <iostream>
#include <utils/nvtx.hpp>

/*
 * CoherentHarmonicFolder performs coherent harmonic summing on the GPU.
 *
 * Unlike incoherent harmonic summing which sums power (losing phase),
 * coherent summing preserves phase information by:
 * 1. Gathering complex Fourier coefficients at harmonic frequencies
 * 2. Searching over initial phase to find optimal alignment
 * 3. Using FFT-based phase search for efficiency
 *
 * This provides optimal sensitivity for ANY duty cycle, but at higher
 * computational cost than incoherent summing.
 *
 * Input: Complex Fourier series (DeviceFourierSeries<cufftComplex>)
 * Output: HarmonicSums (power spectra at each harmonic fold level)
 */
class CoherentHarmonicFolder {
private:
  unsigned int max_blocks;
  unsigned int max_threads;
  float** h_data_ptrs;
  float** d_data_ptrs;
  HarmonicSums<float>& sums;

public:
  CoherentHarmonicFolder(HarmonicSums<float>& sums,
                         unsigned int max_blocks=MAX_BLOCKS,
                         unsigned int max_threads=MAX_THREADS)
    :sums(sums), max_blocks(max_blocks), max_threads(max_threads)
  {
    Utils::device_malloc<float*>(&d_data_ptrs, sums.size());
    Utils::host_malloc<float*>(&h_data_ptrs, sums.size());
  }

  /*
   * Perform coherent harmonic folding on the complex Fourier series.
   *
   * @param fseries The complex Fourier series from FFT
   * @param pspec   The power spectrum (used only for size, output goes to sums)
   *
   * Note: Unlike incoherent folding which operates on power spectrum,
   * coherent folding needs the complex Fourier series to preserve phase.
   */
  void fold(DeviceFourierSeries<cufftComplex>& fseries,
            DevicePowerSpectrum<float>& pspec)
  {
    PUSH_NVTX_RANGE("Coherent Harmonic summing", 2)

    // Set up output pointers for each harmonic fold level
    for (int ii = 0; ii < sums.size(); ii++)
    {
      h_data_ptrs[ii] = sums[ii]->get_data();
    }
    Utils::h2dcpy<float*>(d_data_ptrs, h_data_ptrs, sums.size());

    // Call the coherent harmonic sum kernel
    // It takes the complex Fourier series and outputs to the harmonic sum arrays
    device_coherent_harmonic_sum(fseries.get_data(),
                                  d_data_ptrs,
                                  pspec.get_nbins(),
                                  fseries.get_nbins(),
                                  sums.size(),
                                  max_blocks,
                                  max_threads);

    POP_NVTX_RANGE
  }

  virtual ~CoherentHarmonicFolder()
  {
    Utils::device_free(d_data_ptrs);
    Utils::host_free(h_data_ptrs);
  }
};
