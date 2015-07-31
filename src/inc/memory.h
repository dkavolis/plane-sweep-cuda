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
    static cudaError_t CleanUp(T * ptr)
    {
        if (_onDevice) return cudaFree(ptr);
        else return cudaFreeHost(ptr);
    }

    __host__ __device__ inline
    static cudaError_t Malloc(T * &ptr, size_t & w, size_t & h, size_t &pitch)
    {
        if (_onDevice){
            return cudaMallocPitch(&ptr, &pitch, w * sizeof(T), h);
        }
        else {
            pitch = w * sizeof(T);
            return cudaMallocHost(&ptr, pitch * h);
        }
    }

    __host__ __device__ inline
    static cudaError_t Malloc(T * &ptr, size_t & w, size_t & h, size_t & d, size_t &pitch, size_t &spitch)
    {
        cudaError_t err;
        if (_onDevice){
            err = cudaMallocPitch(&ptr, &pitch, w * sizeof(T), h * d);
            spitch = h * pitch;
        }
        else {
            pitch = w * sizeof(T);
            spitch = h * pitch;
            err = cudaMallocHost(&ptr, pitch * h * d);
        }
        return err;
    }

    // Copy 2D data structures:
    __host__ __device__ inline
    static cudaError_t Device2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyDeviceToDevice);
    }

    __host__ __device__ inline
    static cudaError_t Device2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyDeviceToHost);
    }

    __host__ __device__ inline
    static cudaError_t Host2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyHostToDevice);
    }

    __host__ __device__ inline
    static cudaError_t Host2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyHostToHost);
    }

    // Copy 3D data structures
    __host__ __device__ inline
    static cudaError_t Device2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyDeviceToDevice);
    }

    __host__ __device__ inline
    static cudaError_t Device2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyDeviceToHost);
    }

    __host__ __device__ inline
    static cudaError_t Host2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyHostToDevice);
    }

    __host__ __device__ inline
    static cudaError_t Host2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyHostToHost);
    }
};

#endif // MEMORY_H
