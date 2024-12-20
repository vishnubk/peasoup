#pragma once
#include "cuda.h"
#include "cufft.h"
#include "data_types/timeseries.hpp"
#include "utils/exceptions.hpp"

class CuFFTer {
protected:
  cufftHandle fft_plan;
  unsigned int size; 
  CuFFTer(void):fft_plan(0),size(0){}
  unsigned int get_size(void){return size;}


public:
  double get_resolution(float tsamp){
    return (double) 1.0/(size * tsamp);
  }
  
  virtual unsigned int get_output_size(void){
    return size/2+1;
  }

  virtual ~CuFFTer(){
    if(fft_plan !=0){
      cudaDeviceSynchronize();
      cufftDestroy(fft_plan);
    }
  }

};

class CuFFTerC2C: public CuFFTer {
public:
  CuFFTerC2C(unsigned int size, unsigned int batch=1)
    :CuFFTer()
  {
    this->size = size;
    cufftResult error = cufftPlan1d(&fft_plan, size, CUFFT_C2C, batch);
    ErrorChecker::check_cufft_error(error);
  }
  
  void execute(cufftComplex* input, cufftComplex* output, int direction)
  {
    cufftResult error = cufftExecC2C(fft_plan, input, output, direction);
    ErrorChecker::check_cufft_error(error);
  }
  
  unsigned int get_output_size(void){
    return size;
  }
};

class CuFFTerR2C: public CuFFTer {
public:
  CuFFTerR2C(unsigned int size, long long int batch=1)
    :CuFFTer()
  {
    this->size = size;
    long long int n[] = {static_cast<long long>(size)};
    size_t workSize;
    cufftResult error1 =  cufftCreate(&fft_plan);
    ErrorChecker::check_cufft_error(error1);

    cufftResult error2 = cufftMakePlanMany64(fft_plan, static_cast<long long int>(1), 
      n, 
      NULL, static_cast<long long int>(1), static_cast<long long int>(1), 
      NULL, static_cast<long long int>(1), static_cast<long long int>(size), 
      CUFFT_R2C, 
      static_cast<long long int>(batch), &workSize);
    //cufftResult error = cufftPlan1d(&fft_plan, size, CUFFT_R2C, batch);
    ErrorChecker::check_cufft_error(error2);
  }
  
  void execute(float* tim, cufftComplex* fseries)
  {
    cufftResult error = cufftExecR2C(fft_plan, (cufftReal*) tim, fseries);
    ErrorChecker::check_cufft_error(error);
  }
};

class CuFFTerC2R: public CuFFTer {
public:
  CuFFTerC2R(unsigned int size, long long int batch=1)
    :CuFFTer()
  {
    this->size = size;
    //cufftResult error = cufftPlan1d(&fft_plan, size, CUFFT_C2R, batch);
    long long n[] = {static_cast<long long>(size)};
    size_t workSize;
    cufftResult error1 =  cufftCreate(&fft_plan);
    ErrorChecker::check_cufft_error(error1);

    cufftResult error2 = cufftMakePlanMany64(fft_plan, static_cast<long long int>(1), 
      n, 
      NULL, static_cast<long long int>(1), static_cast<long long int>(1), 
      NULL, static_cast<long long int>(1), static_cast<long long int>(size), 
      CUFFT_C2R, 
      static_cast<long long int>(batch), &workSize);


    ErrorChecker::check_cufft_error(error2);
  }
  
  void execute(cufftComplex* input, float* output)
  {
    cufftResult error = cufftExecC2R(fft_plan, input, (cufftReal*) output);
    ErrorChecker::check_cufft_error(error);
  }
};
