/**
 *  \file memory.h
 *  \brief Header file containing templated class for memory management on GPU / CPU
 */
#ifndef MEMORY_H
#define MEMORY_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

 /**
 *  \brief Templated class for managing memory allocation and deallocation on host / device
 *
 *  \tparam T           type of data to work with
 *  \tparam _onDevice   if data is on device memory set to true, else set to false
 *
 *  \details This is a base class for all other classes that need to manage memory on device and/or host.
 */
template<typename T, bool _onDevice = true>
class MemoryManagement
{
public:

    	/**
     *  \brief Deallocate memory pointed to by \a ptr
     *  
     *  \param ptr pointer to memory to deallocate
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details If \a _onDevice is set to true, pointer has to point to memory on the device, otherwise it has to point
     * to memory on the host.
     */
    __host__ __device__ inline
    static cudaError_t CleanUp(T * ptr)
    {
        if (_onDevice) return cudaFree(ptr);
        else return cudaFreeHost(ptr);
    }

    	/**
     *  \brief Allocate 2D memory and return pointer \a ptr to it
     *  
     *  \param ptr   pointer to allocated memory returned by reference
     *  \param w     width in number of elements
     *  \param h     height in number of elements
     *  \param pitch step size in bytes returned by reference
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details If \a _onDevice is set to true, memory is allocated on the device, else it is allocated
     * on the host.
     */
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

    	/**
     *  \brief This is an overloaded function for 3D memory allocation.
     *  
     *  \param ptr    pointer to allocated memory returned by reference
     *  \param w      width in number of elements
     *  \param h      height in number of elements
     *  \param d      depth in number of elements
     *  \param pitch  step size in bytes returned by reference
     *  \param spitch single slice size in bytes returned by reference
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details Allocate 3D memory and return pointer \a ptr to it.
     *
     * If \a _onDevice is set to true, memory is allocated on the device, else it is allocated
     * on the host.
     */
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
	
    	/**
     *  \brief Device to device 2D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Device2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyDeviceToDevice);
    }

    	/**
     *  \brief Device to host 2D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Device2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyDeviceToHost);
    }

    	/**
     *  \brief Host to device 2D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Host2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyHostToDevice);
    }

    	/**
     *  \brief Host to host 2D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Host2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height, cudaMemcpyHostToHost);
    }
	
    	/**
     *  \brief Device to device 3D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \param depth    depth in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Device2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyDeviceToDevice);
    }

    	/**
     *  \brief Device to host 3D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \param depth    depth in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Device2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyDeviceToHost);
    }

    	/**
     *  \brief Host to device 3D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \param depth    depth in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Host2DeviceCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyHostToDevice);
    }

    	/**
     *  \brief Host to host 3D memory copy
     *  
     *  \param pDst     pointer to destination memory
     *  \param DstPitch step size in bytes of destination memory
     *  \param pSrc     pointer to source memory
     *  \param SrcPitch step size in bytes of source memory
     *  \param width    width in number of elements
     *  \param height   height in number of elements
     *  \param depth    depth in number of elements
     *  \return Returns \a cudaError_t (CUDA error code)
     *  
     *  \details
     */
    __host__ __device__ inline
    static cudaError_t Host2HostCopy(T * &pDst, size_t & DstPitch, T * &pSrc, size_t & SrcPitch, size_t width, size_t height, size_t depth)
    {
        return cudaMemcpy2D(pDst, DstPitch, pSrc, SrcPitch, width * sizeof(T), height * depth, cudaMemcpyHostToHost);
    }
};

#endif // MEMORY_H
