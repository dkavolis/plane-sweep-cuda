#ifndef FUSION_CU_H
#define FUSION_CU_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "fusion.h"



template<unsigned char _bins>
void FusionUpdateU(fusionData<_bins> * f, const double tau, const double lambda, dim3 blocks, dim3 threads);

template<unsigned char _bins>
void FusionUpdateP(fusionData<_bins> * f, const double sigma, dim3 blocks, dim3 threads);

// Missing histogram update step
//template<unsigned char _bins>
//void FusionUpdateIteration(fusionData<_bins> * f, const double tau, const double lambda, const double sigma, dim3 blocks, dim3 threads);

#endif // FUSION_CU_H
