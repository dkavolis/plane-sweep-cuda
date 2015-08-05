/**
 *  \file fusion.h
 *  \brief Header file containing depthmap fusion data class
 */
#ifndef FUSION_H
#define FUSION_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "memory.h"
#include "structs.h"
#include "helper_structs.h"

/** \addtogroup fusion
* @{
*/

/**
 *  \brief Templated class for storing and controlling depthmap fusion data
 *
 *  \tparam _histBins   number of histogram bins
 *  \tparam memT        type of memory, where data is stored
 *
 *  \details This class holds and implements some useful functions to work with depthmap fusion algorithm
 */
template<unsigned char _histBins, MemoryKind memT = Device>
class fusionData : public MemoryManagement<fusionvoxel<_histBins>, memT>, public Managed
{
public:

    /**
     *  \brief Default constructor
     *
     *  \details All size parameters are initialized to 0, no memory allocation takes place
     */
    __device__ __host__ inline
    fusionData() :
        _w(0), _h(0), _d(0), _pitch(0), _spitch(0), _vol(), _own_data(true)
    {
        binParams();
    }

    /**
     *  \brief Overloaded constructor
     *
     *  \param w width in number of voxels
     *  \param h height in number of voxels
     *  \param d depth in number of voxels
     *
     *  \details Allocates memory on call
     */
    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d) :
        _w(w), _h(h), _d(d), _vol(), _own_data(true)
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    /**
     *  \brief Overloaded constructor
     *
     *  \param w width in number of voxels
     *  \param h height in number of voxels
     *  \param d depth in number of voxels
     *  \param x corner of bounding volume in world coordinates
     *  \param y corner opposite to \a x of bounding volume in world coordinates
     *
     *  \details Allocates memory on call
     */
    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d, float3 x, float3 y) :
        _w(w), _h(h), _d(d), _vol(Rectangle3D(x,y)), _own_data(true)
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    /**
     *  \brief Overloaded constructor
     *
     *  \param w   width in number of voxels
     *  \param h   height in number of voxels
     *  \param d   depth in number of voxels
     *  \param vol bounding volume Rectangle3D
     *
     *  \details Allocates memory on call
     */
    __device__ __host__ inline
    fusionData(size_t w, size_t h, size_t d, Rectangle3D& vol) :
        _w(w), _h(h), _d(d), _vol(vol), _own_data(true)
    {
        binParams();
        Malloc(_voxel, _w, _h, _d, _pitch, _spitch);
    }

    __device__ __host__ inline
    fusionData(fusionData<_histBins, memT> & fd) :
        _w(fd.width()), _h(fd.height()), _d(fd.depth()), _pitch(fd.pitch()),
        _spitch(fd.slicePitch()), _vol(fd.volume()), _voxel(fd.voxelPtr()), _own_data(false)
    {
        binParams();
    }

    /**
     *  \brief Default destructor
     *
     *  \details Deallocates memory
     */
    __device__ __host__ inline
    ~fusionData()
    {
        if (_own_data) CleanUp(_voxel);
    }

    // Getters:
    /**
     *  \brief Get width of data
     *
     *  \return Width of data in number of voxel
     *
     *  \details
     */
    __device__ __host__ inline
    size_t width(){ return _w; }

    /**
     *  \brief Get height of data
     *
     *  \return Height of data in number of voxel
     *
     *  \details
     */
    __device__ __host__ inline
    size_t height(){ return _h; }

    /**
     *  \brief Get depth of data
     *
     *  \return Depth of data in number of voxel
     *
     *  \details
     */
    __device__ __host__ inline
    size_t depth(){ return _d; }

    /**
     *  \brief Get pitch of data
     *
     *  \return Step size of data in bytes
     *
     *  \details
     */
    __device__ __host__ inline
    size_t pitch(){ return _pitch; }

    /**
     *  \brief Get slice pitch of data
     *
     *  \return Slice size of data in bytes
     *
     *  \details
     */
    __device__ __host__ inline
    size_t slicePitch(){ return _spitch; }

    /**
     *  \brief Get number of histogram bins
     *
     *  \return Number of histogram bins
     *
     *  \details
     */
    __device__ __host__ inline
    unsigned char bins(){ return _histBins; }

    /**
     *  \brief Get bounding volume rectangle in world coordinates
     *
     *  \return Bounding volume rectangle in world coordinates
     *
     *  \details
     */
    __device__ __host__ inline
    Rectangle3D volume(){ return _vol; }
	
	/**
     *  \brief Get index in the array of element at (x,y,z)
	 *
     *  \return index of the element
     */
	__device__ __host__ inline
	unsigned int index(int nx = 0, int ny = 0, int nz = 0) { return nx+ny*_w+nz*_w*_h; }

    /**
     *  \brief Get number of voxels
     *
     *  \return Number of voxels
     *
     *  \details
     */
    __device__ __host__ inline
    size_t elements(){ return _w * _h * _d; }

    /**
     *  \brief Get data size in bytes
     *
     *  \return Data size in bytes
     *
     *  \details
     */
    __device__ __host__ inline
    size_t sizeBytes(){ return _spitch * _d; }

    /**
     *  \brief Get data size in kilobytes
     *
     *  \return Data size in kilobytes
     *
     *  \details
     */
    __device__ __host__ inline
    double sizeKBytes(){ return sizeBytes() / 1024.f; }

    /**
     *  \brief Get data size in Megabytes
     *
     *  \return Data size in Megabytes
     *
     *  \details
     */
    __device__ __host__ inline
    double sizeMBytes(){ return sizeKBytes() / 1024.f; }

    /**
     *  \brief Get data size in Gigabytes
     *
     *  \return Data size in Gigabytes
     *
     *  \details
     */
    __device__ __host__ inline
    double sizeGBytes(){ return sizeMBytes() / 1024.f; }

    /**
     *  \brief Get world coordinates of voxel center
     *
     *  \param x voxel x index
     *  \param y voxel y index
     *  \param z voxel z index
     *  \return World coordinates
     *
     *  \details
     */
    __device__ __host__ inline
    float3 worldCoords(int x, int y, int z)
    {
        return _vol.a + _vol.size() * make_float3((x + .5) / _w, (y + .5) / _h, (z + .5) / _d);
    }

    // Setters:
    /**
     *  \brief Set bounding volume rectangle in world coordinates
     *
     *  \param vol bounding volume rectangle in world coordinates
     *  \return No return value
     *
     *  \details
     */
    __device__ __host__ inline
    void setVolume(Rectangle3D &vol){ _vol = vol; }

    /**
         *  \brief Set bounding volume rectangle in world coordinates
         *
         *  \param x corner of bounding volume rectangle in world coordinates
         *  \param y opposite corner of bounding volume rectangle in world coordinates
         *  \return No return value
         *
         *  \details
         */
    __device__ __host__ inline
    void setVolume(float3 x, float3 y){ _vol = Rectangle3D(x, y); }

    // Access to elements:
    /**
     *  \brief Access primal to variable \f$u\f$
     *
     *  \param nx voxel x index
     *  \param ny voxel y index
     *  \param nz voxel z index
     *  \return Reference to primal variable \f$u\f$
     *
     *  \details
     */
    __device__ __host__ inline
    float& u(int nx = 0, int ny = 0, int nz = 0)
    {
        return Get(nx, ny, nz).u;
    }

    /**
    *  \brief Constant access to primal variable \f$u\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to primal variable \f$u\f$
    *
    *  \details
    */
    __device__ __host__ inline
    const float& u(int nx = 0, int ny = 0, int nz = 0) const
    {
        return Get(nx, ny, nz).u;
    }

    /**
    *  \brief Access to helper variable \f$v\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to helper variable \f$v\f$
    *
    *  \details
    */
    __device__ __host__ inline
    float& v(int nx = 0, int ny = 0, int nz = 0)
    {
        return Get(nx, ny, nz).v;
    }

    /**
    *  \brief Constant access to helper variable \f$v\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to helper variable \f$v\f$
    *
    *  \details
    */
    __device__ __host__ inline
    const float& v(int nx = 0, int ny = 0, int nz = 0) const
    {
        return Get(nx, ny, nz).v;
    }

    /**
    *  \brief Access to dual variable \f$p\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to dual variable \f$p\f$
    *
    *  \details
    */
    __device__ __host__ inline
    float3& p(int nx = 0, int ny = 0, int nz = 0)
    {
        return Get(nx, ny, nz).p;
    }

    /**
    *  \brief Constant access to dual variable \f$p\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to dual variable \f$p\f$
    *
    *  \details
    */
    __device__ __host__ inline
    const float3& p(int nx = 0, int ny = 0, int nz = 0) const
    {
        return Get(nx, ny, nz).p;
    }

    /**
    *  \brief Access to histogram
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to histogram
    *
    *  \details
    */
    __device__ __host__ inline
    histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0)
    {
        return Get(nx, ny, nz).h;
    }

    /**
    *  \brief Constant access to histogram
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to histogram
    *
    *  \details
    */
    __device__ __host__ inline
    const histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0) const
    {
        return Get(nx, ny, nz).h;
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> * voxelPtr(size_t nz = 0) const
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(_voxel) + nz*_spitch);
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> * voxelPtr(size_t nz = 0)
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(_voxel) + nz*_spitch);
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> * voxelRowPtr(size_t ny = 0, size_t nz = 0)
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(_voxel) + nz*_spitch + ny*_pitch);
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> * voxelRowPtr(size_t ny = 0, size_t nz = 0) const
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(_voxel) + nz*_spitch + ny*_pitch);
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> & operator()(size_t nx = 0, size_t ny = 0, size_t nz = 0)
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> & operator()(size_t nx = 0, size_t ny = 0, size_t nz = 0) const
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> & operator[](size_t ix)
    {
        return _voxel[ix];
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> & operator[](size_t ix) const
    {
        return _voxel[ix];
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> & Get(int nx, int ny, int nz)
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> & Get(int nx, int ny, int nz) const
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    __device__ __host__ inline
    fusionvoxel<_histBins> & Get(int3 p)
    {
        return voxelRowPtr(p.y,p.z)[p.x];
    }

    __device__ __host__ inline
    const fusionvoxel<_histBins> & Get(int3 p) const
    {
        return voxelRowPtr(p.y,p.z)[p.x];
    }

    __device__ __host__ inline
    fusionData<_histBins, memT>& operator=(fusionData<_histBins, memT> & fd)
    {
        if (this == &fd) return *this;
        if (_own_data) CleanUp(_voxel);
        _voxel = fd.voxelPtr();
        _w = fd.width();
        _h = fd.height();
        _d = fd.depth();
        _pitch = fd.pitch();
        _spitch = fd.slicePitch();
        _vol = fd.volume();
        binParams();
        _own_data = false;
        return *this;
    }

    // Get bin parameters:
    /**
     *  \brief Get center of histogram bin
     *
     *  \param binindex index of histogram bin
     *  \return Center of histogram bin
     *
     *  \details
     */
    __device__ __host__ inline
    double binCenter(unsigned char binindex)
    {
        if (binindex < _histBins) return _bincenters[binindex];
        else return 0.f;
    }

    __device__ __host__ inline
    const double * binCenters(){ return &_bincenters[0]; }

    /**
     *  \brief Get distance between histogram centers
     *
     *  \return Distance between histogram centers
     *
     *  \details
     */
    __device__ __host__ inline
    double binStep(){ return _binstep; }

    // Difference functions:
    /**
     *  \brief Get gradient of primal variable \f$u\f$
     *
     *  \param x voxel x index
     *  \param y voxel y index
     *  \param z voxel z index
     *  \return gradient of primal variable \f$u\f$
     *
     *  \details Gradient is calculated from forward differences
     */
    __host__ __device__ inline
    float3 gradUFwd(uint x, uint y, uint z){
        float u = this->u(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->u(x+1, y, z) - u;
        if (y < _h - 1) result.y = this->u(x, y+1, z) - u;
        if (z < _d - 1) result.z = this->u(x, y, z+1) - u;
        return result;
    }

    /**
    *  \brief Get gradient of helper variable \f$v\f$
    *
    *  \param x voxel x index
    *  \param y voxel y index
    *  \param z voxel z index
    *  \return gradient of helper variable \f$v\f$
    *
    *  \details Gradient is calculated from forward differences
    */
    __host__ __device__ inline
    float3 gradVFwd(uint x, uint y, uint z){
        float v = this->v(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < _w - 1) result.x = this->v(x+1, y, z) - v;
        if (y < _h - 1) result.y = this->v(x, y+1, z) - v;
        if (z < _d - 1) result.z = this->v(x, y, z+1) - v;
        return result;
    }

    /**
    *  \brief Get divergence of dual variable \f$p\f$
    *
    *  \param x voxel x index
    *  \param y voxel y index
    *  \param z voxel z index
    *  \return divergence of dual variable \f$p\f$
    *
    *  \details Divergence is calculated from backward differences
    */
    __host__ __device__ inline
    float divPBwd(int x, int y, int z)
    {
        float3 p = this->p(x, y, z);
        float result = p.x + p.y + p.z;
        if (x > 0) result -= this->p(x-1, y, z).x;
        if (y > 0) result -= this->p(x, y-1, z).y;
        if (z > 0) result -= this->p(x, y, z-1).z;
        return result;
    }

    /**
     *  \brief Helper function for \f$\operatorname{prox}_{hist}\f$
     *
     *  \param i index of histogram bin
     *  \param x voxel x index
     *  \param y voxel y index
     *  \param z voxel y index
     *  \return \f$W_i\f$ intermediate variable in \f$\operatorname{prox}_{hist}\f$ calculation
     *
     *  \details
     */
    __host__ __device__ inline
    int Wi(unsigned char i, int x, int y, int z)
    {
        int r = 0;
        for (unsigned char j = 1; j <= i; j++) r -= this->h(x, y, z)(j);
        for (unsigned char j = i + 1; j <= _histBins; j++) r += this->h(x, y, z)(j);
        return r;
    }

    /**
     *  \brief Helper function for \f$\operatorname{prox}_{hist}(u)\f$
     *
     *  \param u      variable in \f$\operatorname{prox}_{hist}(u)\f$
     *  \param i      index of histogram bin
     *  \param x      voxel x index
     *  \param y      voxel y index
     *  \param z      voxel y index
     *  \param tau    depthmap fusion parameter \f$\tau\f$
     *  \param lambda depthmap fusion parameter \f$\lambda\f$
     *  \return \f$p_i\f$ intermediate variable in \f$\operatorname{prox}_{hist}\f$ calculation
     *
     *  \details
     */
    __host__ __device__ inline
    float pi(double u, unsigned char i, int x, int y, int z, double tau, double lambda)
    {
        return u + tau * lambda * Wi(i, x, y, z);
    }

    /**
     *  \brief Calculate \f$\operatorname{prox}_{hist}(u)\f$
     *
     *  \param u      variable in \f$\operatorname{prox}_{hist}(u)\f$
     *  \param x      voxel x index
     *  \param y      voxel y index
     *  \param z      voxel y index
     *  \param tau    depthmap fusion parameter \f$\tau\f$
     *  \param lambda depthmap fusion parameter \f$\lambda\f$
     *  \return Value of \f$\operatorname{prox}_{hist}(u)\f$
     *
     *  \details
     */
    __host__ __device__ inline
    float proxHist(double u, int x, int y, int z, double tau, double lambda)
    {
        sortedHist<_histBins> prox(_bincenters);
        prox.insert(u); // insert p0
        for (unsigned char j = 1; j <= _histBins; j++) prox.insert(pi(u, j, x, y, z, tau, lambda));
        return prox.median();
    }

    /**
     *  \brief Calculate \f$\operatorname{prox}_{\|p\|_{\infty}\le1}(p)\f$
     *
     *  \param x variable \f$p\f$
     *  \return Value of \f$\operatorname{prox}_{\|p\|_{\infty}\le1}(p)\f$
     *
     *  \details
     */
    __host__ __device__ inline
    float3 projectUnitBall(float3 x)
    {
        return x / fmaxf(1.f, sqrt(x.x * x.x + x.y * x.y + x.z * x.z));
    }

    /**
     *  \brief Update voxel histogram
     *
     *  \param x         voxel x index
     *  \param y         voxel y index
     *  \param z         voxel y index
     *  \param voxdepth  voxel depth wrt camera reference frame
     *  \param depth     pixel depth at voxel coordinates interpolated from depthmap
     *  \param threshold signed distance value threshold
     *  \return No return value
     *
     *  \details
     */
    __host__ __device__ inline
    void updateHist(int x, int y, int z, float voxdepth, float depth, float threshold)
    {
        float sd = voxdepth - depth;
        if (_histBins == 2) threshold = 0.f;

        // check if empty
        if (sd >= threshold)
        {
            this->h(x, y, z).last()++;
            return;
        }

        // check if occluded
        if (sd <= - threshold)
        {
            this->h(x, y, z).first()++;
            return;
        }

        // close to surface
        this->h(x, y, z)(roundf((sd + threshold) / (2.f * threshold) * (_histBins - 3)) + 1)++;
    }

    /**
     *  \brief Copy voxel data from host to here
     *
     *  \param data   pointer to source memory
     *  \param npitch step size in bytes of source memory
     *  \return Returns \a cudaError_t (CUDA error code)
     *
     *  \todo Check if it works with managed memory
     */
    __host__ inline
    cudaError_t copyFrom(fusionvoxel<_histBins> * data, size_t npitch)
    {
        if (memT == Device) return Host2DeviceCopy(_voxel, _pitch, data, npitch, _w, _h, _d);
#if CUDA_VERSION_MAJOR >= 6
        if (memT == Managed) return Host2DeviceCopy(_voxel, _pitch, data, npitch, _w, _h, _d);
#endif // CUDA_VERSION_MAJOR >= 6
        return Host2HostCopy(_voxel, _pitch, data, npitch, _w, _h, _d);
    }

    /**
     *  \brief Copy voxel data from here to host
     *
     *  \param data   pointer to destination memory
     *  \param npitch step size in bytes of destination memory
     *  \return Returns \a cudaError_t (CUDA error code)
     *
     *  \todo Check if it works with managed memory
     */
    __host__ inline
    cudaError_t copyTo(fusionvoxel<_histBins> * data, size_t npitch)
    {
        if (memT == Device) return Device2HostCopy(data, npitch, _voxel, _pitch, _w, _h, _d);
#if CUDA_VERSION_MAJOR >= 6
        if (memT == MemoryKind::Managed) return Device2HostCopy(data, npitch, _voxel, _pitch, _w, _h, _d);
#endif // CUDA_VERSION_MAJOR >= 6
        return Host2HostCopy(data, npitch, _voxel, _pitch, _w, _h, _d);
    }

protected:
    /**
    *  \brief Pointer to stored voxel data
    */
    fusionvoxel<_histBins> * _voxel;

    /**
    *  \brief Array of histogram bin center values
    */
    double _bincenters[_histBins];

    /**
    *  \brief Distance between histogram bin centers
    */
    double _binstep;

    /**
    *  \brief Width in number voxels
    */
    size_t  _w;

    /**
    *  \brief Height in number of voxels
    */
    size_t _h;

    /**
    *  \brief Depth in number of voxels
    */
    size_t _d;

    /** \brief Step size in bytes of voxel data */
    size_t _pitch;

    /** \brief Slice size in bytes of voxel data */
    size_t _spitch;

    /** \brief Bounding rectangle in world coordinates */
    Rectangle3D _vol;

    bool _own_data = true;

    /**
     *  \brief Calculate and set histogram bin parameters
     *
     *  \return No return value
     *
     *  \details
     */
    __host__ __device__ inline
    void binParams()
    {
        // index = 0 bin is reserved for occluded voxel (signed distance < -1)
        // index = _histBins - 1 is reserved for empty voxel (signed distance > 1)
        // other bins store signed distance values in the range (-1; 1)
        _bincenters[0] = -1.f;
        _bincenters[_histBins - 1] = 1.f;
        if (_histBins > 3){
            for (unsigned char i = 1; i < _histBins - 1; i++) _bincenters[i] = 2.f * float(i - 1) / float(_histBins - 3) - 1.f;
            _binstep = 2 / float(bins() - 3);
        }
        else {
            if (_histBins == 3) _bincenters[1] = 0.f;
            _binstep = 0.f;
        }
    }
};

/**
*  \brief Convenience typedef for fusionData with 2 bins and stored on device
*/
typedef fusionData<2> dfusionData2;
/**
*  \brief Convenience typedef for fusionData with 3 bins and stored on device
*/
typedef fusionData<3> dfusionData3;
/**
*  \brief Convenience typedef for fusionData with 4 bins and stored on device
*/
typedef fusionData<4> dfusionData4;
/**
*  \brief Convenience typedef for fusionData with 5 bins and stored on device
*/
typedef fusionData<5> dfusionData5;
/**
*  \brief Convenience typedef for fusionData with 6 bins and stored on device
*/
typedef fusionData<6> dfusionData6;
/**
*  \brief Convenience typedef for fusionData with 7 bins and stored on device
*/
typedef fusionData<7> dfusionData7;
/**
*  \brief Convenience typedef for fusionData with 8 bins and stored on device
*/
typedef fusionData<8> dfusionData8;
/**
*  \brief Convenience typedef for fusionData with 9 bins and stored on device
*/
typedef fusionData<9> dfusionData9;
/**
*  \brief Convenience typedef for fusionData with 10 bins and stored on device
*/
typedef fusionData<10> dfusionData10;

/**
*  \brief Convenience typedef for fusionData with 2 bins and stored on host
*/
typedef fusionData<2, Host> fusionData2;
/**
*  \brief Convenience typedef for fusionData with 3 bins and stored on host
*/
typedef fusionData<3, Host> fusionData3;
/**
*  \brief Convenience typedef for fusionData with 4 bins and stored on host
*/
typedef fusionData<4, Host> fusionData4;
/**
*  \brief Convenience typedef for fusionData with 5 bins and stored on host
*/
typedef fusionData<5, Host> fusionData5;
/**
*  \brief Convenience typedef for fusionData with 6 bins and stored on host
*/
typedef fusionData<6, Host> fusionData6;
/**
*  \brief Convenience typedef for fusionData with 7 bins and stored on host
*/
typedef fusionData<7, Host> fusionData7;
/**
*  \brief Convenience typedef for fusionData with 8 bins and stored on host
*/
typedef fusionData<8, Host> fusionData8;
/**
*  \brief Convenience typedef for fusionData with 9 bins and stored on host
*/
typedef fusionData<9, Host> fusionData9;
/**
*  \brief Convenience typedef for fusionData with 10 bins and stored on host
*/
typedef fusionData<10, Host> fusionData10;

/** @} */ // group fusion

#endif // FUSION_H
