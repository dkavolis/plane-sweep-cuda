#ifndef CAM_IMAGE_H
#define CAM_IMAGE_H

#include "image.h"
#include "structs.h"

template<typename T>
struct CamImage : public Image<T, Standard>
{
    Matrix3D R;
    Vector3D t;

    virtual ~CamImage() { free(); }

    template<MemoryKind memT>
    inline __host__ __device__
    CamImage( const Image<T,memT>& img )
        : Image(img), R(), t()
    {}

    inline __host__ __device__
    CamImage( const CamImage<T>& img )
        : Image(img), R(img.R), t(img.t)
    {}

    inline __host__
    CamImage()
        : Image(), R(), t()
    {}

    inline __host__
    CamImage(size_t w, size_t h)
        : Image(w, h), R(), t()
    {
       Malloc(ptr_, w_, h_, pitch_);
    }

    inline __device__ __host__
    CamImage(T* ptr)
        : Image(ptr), R(), t()
    {}

    inline __device__ __host__
    CamImage(T* ptr, size_t w)
        : Image(ptr, w), R(), t()
    {}

    inline __device__ __host__
    CamImage(T* ptr, size_t w, size_t h)
        : Image(ptr, w, h), R(), t()
    {}

    inline __device__ __host__
    CamImage(T* ptr, size_t w, size_t h, size_t pitch)
        : Image(ptr, w, h, pitch), R(), t()
    {}
};

#endif // CAM_IMAGE_H
