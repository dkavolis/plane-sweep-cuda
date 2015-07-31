#ifndef MEMORY_H
#define MEMORY_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

template<typename T, bool _onDevice = true>
class MemoryManagement
{
public:
    __host__ __device__ inline
    void CleanUp(T * ptr)
    {
        if (_onDevice) cudaFree(ptr);
        else cudaFreeHost(ptr);
    }

    __host__ __device__ inline
    void Malloc(T * &ptr, size_t & w, size_t & h, size_t &pitch)
    {
        if (_onDevice){
            cudaMallocPitch(&ptr, &pitch, w * sizeof(T), h);
        }
        else {
            pitch = w * sizeof(T);
            cudaMallocHost(&ptr, pitch * h);
        }
    }

    __host__ __device__ inline
    void Malloc(T * &ptr, size_t & w, size_t & h, size_t & d, size_t &pitch, size_t &spitch)
    {
        if (_onDevice){
            cudaMallocPitch(&ptr, &pitch, w * sizeof(T), h * d);
            spitch = h * pitch;
        }
        else {
            pitch = w * sizeof(T);
            spitch = h * pitch;
            cudaMallocHost(&ptr, pitch * h * d);
        }
    }
};

#endif // MEMORY_H
