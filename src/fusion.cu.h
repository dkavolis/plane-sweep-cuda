#ifndef FUSION_CU_H
#define FUSION_CU_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

typedef unsigned int uint;

// Simple structure to hold histogram data of each voxel
// bin index 0 refers to occluded voxel (signed distance -1)
// bin index _nBins-1 refers to empty voxel (signed distance 1)
// bin signed distance can be calculated as (2 * index / (_nBins - 1) - 1)
template<unsigned int _nBins>
struct histogram
{
    unsigned char bin[_nBins];

    __device__ __host__ histogram()
    {
        for (int i = 0; i < _nBins; i++) bin[i] = 0;
    }

    __device__ __host__ histogram<_nBins>& operator=(histogram<_nBins> & hist)
    {
        if (this == &hist) return *this;
        for (int i = 0; i < _nBins; i++) bin[i] = hist.bin[i];
        return *this;
    }
};

// Simple structure to hold all data of a single voxel 
//required for depthmap fusion
template<unsigned int _nBins>
struct voxel
{
    double u;
    float3 p;
    histogram<_nBins> h;
    
    __host__ __device__ voxel() :
        u(0), p(0, 0, 0), h()
    {}
        
    __host__ __device__ voxel(const double u) :
        this->u(u), p(0, 0, 0), h()
    {}
        
    __host__ __device__ voxel<_nBins>& operator=(voxel<_nBins> & v)
    {
        if (this == &v) return *this;
        u = v.u;
        p = v.p;
        h = v.h;
        return *this;
    }
};

template<unsigned int _histBins, bool _onDevice = true>
class fusionData
{
public:
    __device__ __host__ fusionData() :
        _w(0), _h(0), _d(0), _pitch(0), _spitch(0)
    {
        for (int i = 0; i < _histBins; i++) _bincenters[i] = 2 * float(i) / float(_histBins - 1) - 1.f;
        _binstep = 2 / float(_histBins - 1);
    }

    __device__ __host__ fusionData(size_t w, size_t h, size_t d) :
        _w(w), _h(h), _d(d)
    {
        for (int i = 0; i < _histBins; i++) _bincenters[i] = 2 * float(i) / float(_histBins - 1) - 1.f;
        _binstep = 2 / float(_histBins - 1);
        Malloc<voxel<_histBins>>(_voxel, _pitch, _spitch);
    }

    __device__ __host__ ~fusionData()
    {
        CleanUp<voxel<_histBins>>(_voxel);
    }

    // Getters:
    __device__ __host__ size_t width(){ return _w; }
    __device__ __host__ size_t height(){ return _h; }
    __device__ __host__ size_t depth(){ return _d; }
    __device__ __host__ size_t pitch(){ return _pitch; }
    __device__ __host__ size_t slicePitch(){ return _spitch; }

    __device__ __host__ double u(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _voxel[nx+ny*_w+nz*_w*_h].u;
        else return 0.f;
    }

    __device__ __host__ float3 p(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _voxel[nx+ny*_w+nz*_w*_h].p;
        else return make_float3(0, 0, 0);
    }

    __device__ __host__ histogram<_histBins> h(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _voxel[nx+ny*_w+nz*_w*_h].h;
        else return histogram<_histBins>();
    }

    __device__ __host__ voxel<_histBins> * voxelPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return &_voxel[nx+ny*_w+nz*_w*_h];
        else return nullptr;
    }

    __device__ __host__ float binCenter(int binindex)
    {
        if ((binindex >= 0) && (binindex < _histBins)) return _bincenters[binindex];
        else return 0.f;
    }

    __device__ __host__ float binStep(){ return _binstep; }

//    __device__ __host__ void uPtr(float * u){ _u = u; }
//    __device__ __host__ void pPtr(float2 * p){ _p = p; }
//    __device__ __host__ void histPtr(histogram<_histBins> * h){ _hst = h; }

    __host__ __device__ float3 gradUFwd(uint x, uint y, uint z){
        float u = this->u(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->u(x+1, y, z) - u;
        if (y < _h - 1) result.y = this->u(x, y+1, z) - u;
        if (z < _d - 1) result.z = this->u(x, y, z+1) - u;
        return result;
    }

    __host__ __device__ float divPBwd(int x, int y, int z)
    {
        float3 p = this->p(x, y, z);
        float result = p.x + p.y + p.z;
        if (x > 0) result -= this->p(x-1, y, z).x;
        if (y > 0) result -= this->p(x, y-1, z).y;
        if (z > 0) result -= this->p(x, y, z-1).z;
        return result;
    }

protected:
    voxel<_histBins> * _voxel;
    float _bincenters[_histBins], _binstep;
    size_t  _w, _h, _d, _pitch, _spitch;

    template<typename T>
    __host__ __device__ void CleanUp(T * ptr)
    { 
        if (_onDevice) cudaFree(ptr); 
        else cudaFreeHost(ptr);
    }

    template<typename T>
    __host__ __device__ void Malloc(T * &ptr, size_t &pitch, size_t &spitch)
    {
        if (_onDevice){
            cudaMallocPitch(&ptr, &pitch, _w * sizeof(T), _h * _d);
            spitch = _h * pitch;
        }
        else {
            pitch = _w * sizeof(T);
            spitch = _h * pitch;
            cudaMallocHost(&ptr, pitch * _h * _d);
        }
    }
};

// typedefs for fusionData with memory on GPU
typedef fusionData<2> dfusionData2;
typedef fusionData<3> dfusionData3;
typedef fusionData<4> dfusionData4;
typedef fusionData<5> dfusionData5;
typedef fusionData<6> dfusionData6;
typedef fusionData<7> dfusionData7;
typedef fusionData<8> dfusionData8;
typedef fusionData<9> dfusionData9;
typedef fusionData<10> dfusionData10;

// typedefs for fusionData with memory on CPU
typedef fusionData<2, false> fusionData2;
typedef fusionData<3, false> fusionData3;
typedef fusionData<4, false> fusionData4;
typedef fusionData<5, false> fusionData5;
typedef fusionData<6, false> fusionData6;
typedef fusionData<7, false> fusionData7;
typedef fusionData<8, false> fusionData8;
typedef fusionData<9, false> fusionData9;
typedef fusionData<10, false> fusionData10;

#endif // FUSION_CU_H
