#pragma once
#include <data_types/timeseries.hpp>
#include <kernels/kernels.h>
#include <kernels/defaults.h>
#include <utils/exceptions.hpp>

class TimeDomainResampler {
private:
  unsigned int max_threads;
  unsigned int max_blocks;
  
public:
  TimeDomainResampler(unsigned int max_threads=MAX_THREADS, unsigned int max_blocks=MAX_BLOCKS)
    :max_threads(max_threads),max_blocks(max_blocks)    
  {
  }
  
  //Force float until the kernel gets templated
  void resample(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, 
		unsigned int size, float acc)
  {
    device_resample(input.get_data(), output.get_data(), size,
		    acc, input.get_tsamp(),max_threads,  max_blocks);
  }

  void resampleII(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output,
                unsigned int size, float acc)
  {
    device_resampleII(input.get_data(), output.get_data(), size,
                    acc, input.get_tsamp(),max_threads,  max_blocks);
  }

  void circular_orbit_resampler(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, unsigned int size, double n, double a1, double phi, double tsamp, double inverse_tsamp)
  {
    //double zero_offset = a1 * sin(phi) * inverse_tsamp;
    double zero_offset = a1 * -1 * sin(phi) * inverse_tsamp;
    device_circular_orbit_resampler(input.get_data(), output.get_data(), n, a1, phi, zero_offset, tsamp, inverse_tsamp, size, max_threads, max_blocks);
  }

  void elliptical_orbit_resampler_approx(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, unsigned int size, double n, double a1, double phi, double omega, double ecc, double tsamp, double inverse_tsamp)
  {
   device_elliptical_orbit_resampler_approx(input.get_data(), output.get_data(), n, a1, phi, omega, ecc, tsamp, inverse_tsamp, size, max_threads, max_blocks);
}

//void remove_roemer_delay_elliptical_exact(double* start_timeseries_array, double* roemer_delay_removed_timeseries_array, unsigned int size,\
//   double n, double a1, double phi_n, double omega, double ecc, double tsamp)
void remove_roemer_delay_elliptical_exact(DeviceTimeSeries<float>& input, DeviceTimeSeries<float>& output, unsigned int size,\
   double n, double a1, double phi_n, double omega, double ecc, double tsamp)

   {
   //device_remove_roemer_delay_elliptical_exact(start_timeseries_array, roemer_delay_removed_timeseries_array, n, a1, phi_n, omega, ecc, tsamp, size, max_threads, max_blocks);
   device_remove_roemer_delay_elliptical_exact(input.get_data(), output.get_data(), n, a1, phi_n, omega, ecc, tsamp, size, max_threads, max_blocks);

   }

// void resample_using_1D_lerp(double* roemer_delay_removed_timeseries_array, DeviceTimeSeries<float>& input,  unsigned long xp_len, 
//     unsigned long x_len, double* output_samples_array, DeviceTimeSeries<float>& output)

//    {
   
//    device_resample_using_1D_lerp(roemer_delay_removed_timeseries_array, input.get_data(), xp_len, x_len, output_samples_array, output.get_data(), max_threads, max_blocks);
//  }



};


