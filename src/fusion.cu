// Kernels for depthmap fusion (WIP):
#include "fusion.cu.h"

template<unsigned char _bins>
__global__ void FusionUpdateHistogram_kernel(fusionData<_bins> * f, const float * __restrict__ depthmap, const float threshold)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {

    }
}

template<unsigned char _bins>
__global__ void FusionUpdateU_kernel(fusionData<_bins> * f, const double tau, const double lambda)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {
        const double un = f->u(x, y, z);
        const double u = un - tau * (- f->divPBwd(x, y, z));
        f->u(x, y, z) = f->proxHist(u, x, y, z, tau, lambda);
        f->v(x, y, z) = 2 * f->u(x, y, z) - un;
    }
}

template<unsigned char _bins>
__global__ void FusionUpdateP_kernel(fusionData<_bins> * f, const double sigma)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    if ((x < f->width()) && (y < f->height()) && (z < f->depth()))
    {
        f->p(x, y, z) = f->projectUnitBall(f->p(x, y, z) + sigma * f->gradVFwd(x, y, z));
    }
}

template<unsigned char _bins>
void FusionUpdateU(fusionData<_bins> * f, const double tau, const double lambda, dim3 blocks, dim3 threads)
{
    FusionUpdateU_kernel<_bins><<<blocks, threads>>>(f, tau, lambda);
}

template<unsigned char _bins>
void FusionUpdateP(fusionData<_bins> * f, const double sigma, dim3 blocks, dim3 threads)
{
    FusionUpdateP_kernel<_bins><<<blocks, threads>>>(f, sigma);
}

// Missing histogram update step
//template<unsigned char _bins>
//void FusionUpdateIteration(fusionData<_bins> * f, const double tau, const double lambda, const double sigma, dim3 blocks, dim3 threads)
//{
//    FusionUpdateU_kernel<_bins><<<blocks, threads>>>(f, tau, lambda);
//    FusionUpdateP_kernel<_bins><<<blocks, threads>>>(f, sigma);
//}
