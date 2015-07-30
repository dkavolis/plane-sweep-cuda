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

template<unsigned int _histBins, bool _onDevice = true>
class fusionData
{
public:
    __device__ __host__ fusionData() :
        _w(0), _h(0), _d(0), _pitchu(0), _spitchu(0), _pitchp(0), _spitchp(0), _pitchh(0), _spitchh(0)
    {
        for (int i = 0; i < _histBins; i++) _bincenters[i] = 2 * float(i) / float(_histBins - 1) - 1.f;
        _binstep = 2 / float(_histBins - 1);
    }

    __device__ __host__ fusionData(size_t w, size_t h, size_t d) :
        _w(w), _h(h), _d(d)
    {
        for (int i = 0; i < _histBins; i++) _bincenters[i] = 2 * float(i) / float(_histBins - 1) - 1.f;
        _binstep = 2 / float(_histBins - 1);
        Malloc<typeof(_u)>(_u, _pitchu, _spitchu);
        Malloc<typeof(_p)>(_p, _pitchp, _spitchp);
        Malloc<typeof(_hst)>(_hst, _pitchh, _spitchh);
    }

    __device__ __host__ ~fusionData()
    {
        CleanUp<typeof(_u)>(_u);
        CleanUp<typeof(_p)>(_p);
        CleanUp<typeof(_hst)>(_hst);
    }

    // Getters:
    __device__ __host__ size_t width(){ return _w; }
    __device__ __host__ size_t height(){ return _h; }
    __device__ __host__ size_t depth(){ return _d; }
    __device__ __host__ size_t pitchU(){ return _pitchu; }
    __device__ __host__ size_t slicePitchU(){ return _spitchu; }
    __device__ __host__ size_t pitchP(){ return _pitchp; }
    __device__ __host__ size_t slicePitchP(){ return _spitchp; }
    __device__ __host__ size_t pitchHist(){ return _pitchh; }
    __device__ __host__ size_t slicePitchHist(){ return _spitchh; }

    __device__ __host__ float u(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _u[nx+ny*_w+nz*_w*_h];
        else return 0.f;
    }

    __device__ __host__ float3 p(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _p[nx+ny*_w+nz*_w*_h];
        else return make_float3(0, 0, 0);
    }

    __device__ __host__ histogram<_histBins> hist(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return _hst[nx+ny*_w+nz*_w*_h];
        else return histogram<_histBins>();
    }

    __device__ __host__ float * uPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return &_u[nx+ny*_w+nz*_w*_h];
        else return nullptr;
    }

    __device__ __host__ float3 * pPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return &_p[nx+ny*_w+nz*_w*_h];
        else return nullptr;
    }

    __device__ __host__ histogram<_histBins> * histPtr(int nx = 0, int ny = 0, int nz = 0)
    {
        if ((nx<_w)&&(ny<_h)&&(nz<_d)&&(nx>=0)&&(ny>=0)&&(nz>=0)) return &_hst[nx+ny*_w+nz*_w*_h];
        else return nullptr;
    }

    __device__ __host__ float binCenter(int binindex)
    {
        if ((binindex >= 0) && (binindex < _histBins)) return _bincenters[binindex];
        else return 0.f;
    }

    __device__ __host__ float binStep(){ return _binstep; }

    __device__ __host__ void uPtr(float * u){ _u = u; }
    __device__ __host__ void pPtr(float2 * p){ _p = p; }
    __device__ __host__ void histPtr(histogram<_histBins> * h){ _hst = h; }

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
    float * _u;
    float3 * _p;
    histogram<_histBins> * _hst;
    float _bincenters[_histBins], _binstep;
    size_t  _w, _h, _d,
            _pitchu, _spitchu,
            _pitchp, _spitchp,
            _pitchh, _spitchh;

    template<typename T>
    __host__ __device__ void CleanUp(T * ptr){ if (_onDevice) cudaFree(ptr); else cudaFreeHost(ptr); }

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
typedef fusionData<2, true> dfusionData2;
typedef fusionData<3, true> dfusionData3;
typedef fusionData<4, true> dfusionData4;
typedef fusionData<5, true> dfusionData5;
typedef fusionData<6, true> dfusionData6;
typedef fusionData<7, true> dfusionData7;
typedef fusionData<8, true> dfusionData8;
typedef fusionData<9, true> dfusionData9;
typedef fusionData<10, true> dfusionData10;

// typedefs for fusionData with memory on CPU
typedef fusionData<2> fusionData2;
typedef fusionData<3> fusionData3;
typedef fusionData<4> fusionData4;
typedef fusionData<5> fusionData5;
typedef fusionData<6> fusionData6;
typedef fusionData<7> fusionData7;
typedef fusionData<8> fusionData8;
typedef fusionData<9> fusionData9;
typedef fusionData<10> fusionData10;

#endif // FUSION_CU_H
