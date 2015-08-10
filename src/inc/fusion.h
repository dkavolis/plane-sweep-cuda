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

// Forward declarations
template<unsigned char _histBins, MemoryKind memT = Device>
class fusionData;

template<unsigned char bins>
struct fusionDataSettings;

/** \addtogroup fusion
* @{
*/

/**
 *  \brief Structure for storing all fusionData settings
 *
 *  \tparam _bins   number of histogram bins
 *
 *  \details Required as a workaround around what appears to be \a cudaMemcpy bug. fusionData copied to the device
 *  contains garbage member values, which seem to be shifted wrong adresses on the device memory. Copying back with garbage values
 *  returns original values.
 *
 *  * Wrapping up all fusionData members in this structure restores correct \a cudaMemcpy function.
 */
template<unsigned char _bins>
struct fusionDataSettings : public Manage
{
    size_t  width,
            height,
            depth,
            pitch,
            spitch;
    float binstep;
    Rectangle3D volume;
    float bin[_bins];
    bool own;
    fusionvoxel<_bins> * ptr;

    __host__ __device__ inline
    fusionDataSettings() :
        width(0), height(0), depth(0), pitch(0), spitch(0), binstep(0), volume(), own(false), ptr(0)
    {}

    __host__ __device__ inline
    fusionDataSettings(size_t width, size_t height, size_t depth, size_t pitch, size_t spitch, float step, bool manage,
                       fusionvoxel<_bins> * pointer, Rectangle3D volume) :
        width(width), height(height), depth(depth), pitch(pitch), spitch(spitch),
        binstep(step), volume(volume), own(manage), ptr(pointer)
    {}

    __host__ __device__ inline
    fusionDataSettings(const fusionDataSettings<_bins> & f) :
        width(f.width), height(f.height), depth(f.depth), pitch(f.pitch), spitch(f.spitch),
        binstep(f.binstep), volume(f.volume), own(f.own), ptr(f.ptr)
    {
        for (int i = 0; i < _bins; i++) bin[i] = f.bin[i];
    }

    __host__ __device__ inline
    fusionDataSettings(fusionData<_bins> & f) :
        width(f.width()), height(f.height()), depth(f.depth()), pitch(f.pitch()), spitch(f.slicePitch()),
        binstep(f.binStep()), volume(f.volume()), own(f.ManageData()), ptr(f.voxelPtr())
    {
        for (int i = 0; i < _bins; i++) bin[i] = f.binCenter(i);
    }

    __host__ __device__ inline
    fusionDataSettings<_bins>& operator=(const fusionDataSettings<_bins> & f)
    {
        if (this == &f) return *this;
        width = f.width;
        height = f.height;
        depth = f.depth;
        pitch = f.pitch;
        spitch = f.spitch;
        binstep = f.binstep;
        volume = f.volume;
        own = f.own;
        ptr = f.ptr;
        for (int i = 0; i < _bins; i++) bin[i] = f.bin[i];
        return *this;
    }
};

/**
 *  \brief Templated class for storing and controlling depthmap fusion data
 *
 *  \tparam _histBins   number of histogram bins
 *  \tparam memT        type of memory, where data is stored
 *
 *  \details This class holds and implements some useful functions to work with depthmap fusion algorithm
 */
template<unsigned char _histBins, MemoryKind memT>
class fusionData : public MemoryManagement<fusionvoxel<_histBins>, memT>, public Manage
{
public:

    /**
     *  \brief Default constructor
     *
     *  \details All size parameters are initialized to 0, no memory allocation takes place
     */
    __device__ __host__ inline
    fusionData() :
      stg(0, 0, 0, 0, 0, 0, true, 0, Rectangle3D())
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
        stg(w, h, d, 0, 0, 0, true, 0, Rectangle3D())
    {
        binParams();
        Malloc(stg.ptr, stg.width, stg.height, stg.depth, stg.pitch, stg.spitch);
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
        stg(w, h, d, 0, 0, 0, true, 0, Rectangle3D(x, y))
    {
        binParams();
        Malloc(stg.ptr, stg.width, stg.height, stg.depth, stg.pitch, stg.spitch);
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
        stg(w, h, d, 0, 0, 0, true, 0, vol)
    {
        binParams();
        Malloc(stg.ptr, stg.width, stg.height, stg.depth, stg.pitch, stg.spitch);
    }

    __device__ __host__ inline
    fusionData(const fusionData<_histBins, memT> & fd)
    {
        stg = fd.exportSettings();
        stg.own = false;
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
        if (stg.own) CleanUp(stg.ptr);
    }

    // Getters:
    /**
     *  \brief Get width of data
     *
     *  \return Width of data in number of voxel
     */
    __device__ __host__ inline
    size_t width(){
        return stg.width;
    }

    /**
     *  \brief Get height of data
     *
     *  \return Height of data in number of voxel
     */
    __device__ __host__ inline
    size_t height(){
        return stg.height;
    }

    /**
     *  \brief Get depth of data
     *
     *  \return Depth of data in number of voxel
     */
    __device__ __host__ inline
    size_t depth(){
        return stg.depth;
    }

    /**
     *  \brief Get pitch of data
     *
     *  \return Step size of data in bytes
     */
    __device__ __host__ inline
    size_t pitch(){
        return stg.pitch;
    }

    /**
     *  \brief Get slice pitch of data
     *
     *  \return Slice size of data in bytes
     */
    __device__ __host__ inline
    size_t slicePitch(){
        return stg.spitch;
    }

    /**
     *  \brief Get number of histogram bins
     *
     *  \return Number of histogram bins
     */
    __device__ __host__ inline
    unsigned char bins(){ return _histBins; }

    /**
     *  \brief Get bounding volume rectangle in world coordinates
     *
     *  \return Bounding volume rectangle in world coordinates
     */
    __device__ __host__ inline
    Rectangle3D volume(){
        return stg.volume;
    }
	
	/**
     *  \brief Get index in the array of element at (x,y,z)
	 *
     *  \return index of the element
     */
	__device__ __host__ inline
    int index(int nx = 0, int ny = 0, int nz = 0) {
        return nx+ny*stg.width+nz*stg.width*stg.height;
    }

    /**
     *  \brief Get 3D indexes from array index
     *
     *  \return (x,y,z) indexes of the element in 3D array
     */
    __device__ __host__ inline
    int3 indexes(int ix)
    {
        int3 r;
        r.x = ix % stg.width;
        r.y = ix / stg.width % stg.height;
        r.z = ix / (stg.width * stg.height);
        return r;
    }

    /**
     *  \brief Resize voxel data
     *
     *  \param w    new width
     *  \param h    new height
     *  \param d    new depth
     *  \return If data is managed it is deallocated. In any case, the new data will be managed.
     */
    __host__ inline
    void Resize(size_t w, size_t h, size_t d)
    {
        stg.width = w;
        stg.height = h;
        stg.depth = d;
        if (stg.own) CleanUp(stg.ptr);
        stg.own = true;
        Malloc(stg.ptr, stg.width, stg.height, stg.depth, stg.pitch, stg.spitch);
    }

    /**
     *  \brief Get number of voxels
     *
     *  \return Number of voxels
     */
    __device__ __host__ inline
    size_t elements(){
        return stg.width * stg.height * stg.depth;
    }

    /**
     *  \brief Get data size in bytes
     *
     *  \return Data size in bytes
     */
    __device__ __host__ inline
    size_t sizeBytes(){
        return stg.spitch * stg.depth;
    }

    /**
     *  \brief Get data size in kilobytes
     *
     *  \return Data size in kilobytes
     */
    __device__ __host__ inline
    float sizeKBytes(){ return sizeBytes() / 1024.f; }

    /**
     *  \brief Get data size in Megabytes
     *
     *  \return Data size in Megabytes
     */
    __device__ __host__ inline
    float sizeMBytes(){ return sizeKBytes() / 1024.f; }

    /**
     *  \brief Get data size in Gigabytes
     *
     *  \return Data size in Gigabytes
     */
    __device__ __host__ inline
    float sizeGBytes(){ return sizeMBytes() / 1024.f; }

    /**
     *  \brief Get world coordinates of voxel center
     *
     *  \param x voxel x index
     *  \param y voxel y index
     *  \param z voxel z index
     *  \return World coordinates
     */
    __device__ __host__ inline
    float3 worldCoords(int x, int y, int z)
    {
        return stg.volume.a + stg.volume.size() * make_float3((x + .5) / stg.width, (y + .5) / stg.height, (z + .5) / stg.depth);
    }

    /**
     *  \brief Get state of data stored
     *
     *  \return True - data will be deleted when destructor is called
     *          False - data will be kept on destructor call
     */
    __device__ __host__ inline
    bool ManageData(){
        return stg.own;
    }

    // Setters:
    /**
     *  \brief Set state of data stored
     *
     *  \param manage   Set true if data should be deleted on destructor call
     *                  Set false if data should be kept on destructor call
     *  \return No return value
     */
    __device__ __host__ inline
    void setManageData(bool manage){
        stg.own = manage;
    }

    /**
     *  \brief Set bounding volume rectangle in world coordinates
     *
     *  \param vol bounding volume rectangle in world coordinates
     *  \return No return value
     */
    __device__ __host__ inline
    void setVolume(Rectangle3D &vol){
        stg.volume = vol;
    }

    /**
         *  \brief Set bounding volume rectangle in world coordinates
         *
         *  \param x corner of bounding volume rectangle in world coordinates
         *  \param y opposite corner of bounding volume rectangle in world coordinates
         *  \return No return value
         */
    __device__ __host__ inline
    void setVolume(float3 x, float3 y){
        stg.volume = Rectangle3D(x, y);
    }

    // Access to elements:
    /**
     *  \brief Access primal to variable \f$u\f$
     *
     *  \param nx voxel x index
     *  \param ny voxel y index
     *  \param nz voxel z index
     *  \return Reference to primal variable \f$u\f$
     */
    __device__ __host__ inline
    float& u(int nx = 0, int ny = 0, int nz = 0)
    {
        return voxelRowPtr(ny, nz)[nx].u;
    }

    /**
    *  \brief Constant access to primal variable \f$u\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to primal variable \f$u\f$
    */
    __device__ __host__ inline
    const float& u(int nx = 0, int ny = 0, int nz = 0) const
    {
        return voxelRowPtr(ny, nz)[nx].u;
    }

    /**
    *  \brief Access to helper variable \f$v\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to helper variable \f$v\f$
    */
    __device__ __host__ inline
    float& v(int nx = 0, int ny = 0, int nz = 0)
    {
        return voxelRowPtr(ny, nz)[nx].v;
    }

    /**
    *  \brief Constant access to helper variable \f$v\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to helper variable \f$v\f$
    */
    __device__ __host__ inline
    const float& v(int nx = 0, int ny = 0, int nz = 0) const
    {
        return voxelRowPtr(ny, nz)[nx].v;
    }

    /**
    *  \brief Access to dual variable \f$p\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to dual variable \f$p\f$
    */
    __device__ __host__ inline
    float3& p(int nx = 0, int ny = 0, int nz = 0)
    {
        return voxelRowPtr(ny, nz)[nx].p;
    }

    /**
    *  \brief Constant access to dual variable \f$p\f$
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to dual variable \f$p\f$
    */
    __device__ __host__ inline
    const float3& p(int nx = 0, int ny = 0, int nz = 0) const
    {
        return voxelRowPtr(ny, nz)[nx].p;
    }

    /**
    *  \brief Access to histogram
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Reference to histogram
    */
    __device__ __host__ inline
    histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0)
    {
        return voxelRowPtr(ny, nz)[nx].h;
    }

    /**
    *  \brief Constant access to histogram
    *
    *  \param nx voxel x index
    *  \param ny voxel y index
    *  \param nz voxel z index
    *  \return Constant reference to histogram
    */
    __device__ __host__ inline
    const histogram<_histBins>& h(int nx = 0, int ny = 0, int nz = 0) const
    {
        return voxelRowPtr(ny, nz)[nx].h;
    }

    /**
     *  \brief Get const pointer to voxel data
     *
     *  \param nz plane of first element in voxel data
     *  \return Const pointer to first element in plane \p nz
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> * voxelPtr(int nz = 0) const
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(stg.ptr) + nz*stg.spitch);
    }

    /**
     *  \brief Get pointer to voxel data
     *
     *  \param nz plane of first element in voxel data
     *  \return Pointer to first element in plane \p nz
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> * voxelPtr(int nz = 0)
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(stg.ptr) + nz*stg.spitch);
    }

    /**
     *  \brief Get pointer to voxel data
     *
     *  \param ny row index
     *  \param nz plane index
     *  \return Pointer to first element in row \p ny and plane \p nz
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> * voxelRowPtr(int ny = 0, int nz = 0)
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(stg.ptr) + nz*stg.spitch + ny*stg.pitch);
    }

    /**
     *  \brief Get const pointer to voxel data
     *
     *  \param ny row index
     *  \param nz plane index
     *  \return Const pointer to first element in row \p ny and plane \p nz
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> * voxelRowPtr(int ny = 0, int nz = 0) const
    {
        return (fusionvoxel<_histBins> *)((unsigned char*)(stg.ptr) + nz*stg.spitch + ny*stg.pitch);
    }

    /**
     *  \brief Access operator
     *
     *  \param nx column index
     *  \param ny row index
     *  \param nz plane index
     *  \return Reference to element in column \p nx, row \p ny and plane \p nz.
     *
     *  \details
     *  \p memT has to be other than \p Device for this function to work on host.
     *  \p memT has to be either \p Device or \p Managed for this function to work on kernel.
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> & operator()(int nx = 0, int ny = 0, int nz = 0)
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    /**
     *  \brief Const access operator
     *
     *  \param nx column index
     *  \param ny row index
     *  \param nz plane index
     *  \return Const reference to element in column \p nx, row \p ny and plane \p nz.
     *
     *  \details
     *  \p memT has to be other than \p Device for this function to work on host.
     *  \p memT has to be either \p Device or \p Managed for this function to work on kernel.
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> & operator()(int nx = 0, int ny = 0, int nz = 0) const
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    /**
     *  \brief Access operator
     *
     *  \param ix voxel index
     *  \return Reference to element with index \p ix
     *
     *  \details
     *  \p memT has to be other than \p Device for this function to work on host.
     *  \p memT has to be either \p Device or \p Managed for this function to work on kernel.
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> & operator[](int ix)
    {
        return stg.ptr[ix];
    }

    /**
     *  \brief Const access operator
     *
     *  \param ix voxel index
     *  \return Const reference to element with index \p ix
     *
     *  \details
     *  \p memT has to be other than \p Device for this function to work on host.
     *  \p memT has to be either \p Device or \p Managed for this function to work on kernel.
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> & operator[](int ix) const
    {
        return stg.ptr[ix];
    }

    /**
     *  \brief Access operator
     *
     *  \param ix voxel index
     *  \return Reference to element with index \p ix
     *
     *  \details
     *  \p memT has to be other than \p Device for this function to work on host.
     *  \p memT has to be either \p Device or \p Managed for this function to work on kernel.
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> & Get(int nx, int ny, int nz)
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    /**
     *  \brief Same as fusionData::operator()
     *
     *  \sa fusionData::operator()
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> & Get(int nx, int ny, int nz) const
    {
        return voxelRowPtr(ny,nz)[nx];
    }

    /**
     *  \brief Same as fusionData::operator()
     *
     *  \param p    indexes of the voxel
     *  \sa fusionData::operator()
     */
    __device__ __host__ inline
    fusionvoxel<_histBins> & Get(int3 p)
    {
        return voxelRowPtr(p.y,p.z)[p.x];
    }

    /**
     *  \brief Same as fusionData::operator()
     *
     *  \param p    indexes of the voxel
     *  \sa fusionData::operator()
     */
    __device__ __host__ inline
    const fusionvoxel<_histBins> & Get(int3 p) const
    {
        return voxelRowPtr(p.y,p.z)[p.x];
    }

    /**
     *  \brief Assignment operator
     *
     *  \details Does not perform deep copy of the data and does not manage it.
     */
    __device__ __host__ inline
    fusionData<_histBins, memT>& operator=(fusionData<_histBins, memT> & fd)
    {
        if (this == &fd) return *this;
        if (stg.own) CleanUp(stg.ptr);
        stg = fd.exportSettings();
        binParams();
        stg.own = false;
        return *this;
    }

    // Get bin parameters:
    /**
     *  \brief Get center of histogram bin
     *
     *  \param binindex index of histogram bin
     *  \return Center of histogram bin
     */
    __device__ __host__ inline
    float binCenter(unsigned char binindex)
    {
        if (binindex < _histBins) return stg.bin[binindex];
        else return 0.f;
    }

    /**
     *  \brief Get pointer to array of bin centers
     */
    __device__ __host__ inline
    const float * binCenters(){
        return &stg.bin[0];
    }

    /**
     *  \brief Get distance between histogram centers
     *
     *  \return Distance between histogram centers
     */
    __device__ __host__ inline
    float binStep(){
        return stg.binstep;
    }

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
    float3 gradUFwd(int x, int y, int z){
        float u = this->u(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < stg.width - 1) result.x = this->u(x+1, y, z) - u;
        if (y < stg.height - 1) result.y = this->u(x, y+1, z) - u;
        if (z < stg.depth - 1) result.z = this->u(x, y, z+1) - u;
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
    float3 gradVFwd(int x, int y, int z){
        float v = this->v(x, y, z);
        float3 result = make_float3(0.f, 0.f, 0.f);
        if (x < stg.width - 1) result.x = this->v(x+1, y, z) - v;
        if (y < stg.height - 1) result.y = this->v(x, y+1, z) - v;
        if (z < stg.depth - 1) result.z = this->v(x, y, z+1) - v;
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
     */
    __host__ __device__ inline
    int Wi(unsigned char i, int x, int y, int z)
    {
        int r = 0;
        for (unsigned char j = 1; j <= i; j++) r -= this->h(x, y, z)(j-1);
        for (unsigned char j = i + 1; j <= _histBins; j++) r += this->h(x, y, z)(j-1);
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
     */
    __host__ __device__ inline
    float pi(float u, unsigned char i, int x, int y, int z, float tau, float lambda)
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
     */
    __host__ __device__ inline
    float proxHist(float u, int x, int y, int z, float tau, float lambda)
    {
        sortedHist<_histBins> prox(stg.bin);
        prox.insert(u); // insert p0
        for (unsigned char j = 1; j <= _histBins; j++) prox.insert(pi(u, j, x, y, z, tau, lambda));
        return prox.median();
    }

    /**
     *  \brief Calculate \f$\operatorname{prox}_{\|p\|_{\infty}\le1}(p)\f$
     *
     *  \param x variable \f$p\f$
     *  \return Value of \f$\operatorname{prox}_{\|p\|_{\infty}\le1}(p)\f$
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
     */
    __host__ __device__ inline
    void updateHist(int x, int y, int z, float voxdepth, float depth, float threshold)
    {
        float sd = depth - voxdepth;
        float thresh = threshold;
        if (_histBins == 2) thresh = 0.f;

        // check if empty
        if (sd >= thresh)
        {
            this->h(x, y, z).last()++;
            return;
        }

        // check if occluded
        if (sd <= - thresh)
        {
            this->h(x, y, z).first()++;
            return;
        }

        // close to surface
        unsigned char i = (unsigned char)fminf(roundf((sd + thresh) / (2.f * thresh) * (_histBins - 3) + 1), _histBins - 2);
        this->h(x, y, z).bin[i]++;
    }

    /**
     *  \brief Copy voxel data from host to here
     *
     *  \param data   pointer to source memory
     *  \param npitch step size in bytes of source memory
     *  \return Returns \a cudaError_t (CUDA error code)
     */
    __host__ inline
    cudaError_t copyFrom(fusionvoxel<_histBins> * data, size_t npitch)
    {
        if (memT == Device) return Host2DeviceCopy(stg.ptr, stg.pitch, data, npitch, stg.width, stg.height, stg.depth);
#if CUDA_VERSION_MAJOR >= 6
        if (memT == Managed) return Host2DeviceCopy(stg.ptr, stg.pitch, data, npitch, stg.width, stg.height, stg.depth);
#endif // CUDA_VERSION_MAJOR >= 6
        return Host2HostCopy(stg.ptr, stg.pitch, data, npitch, stg.width, stg.height, stg.depth);
    }

    /**
     *  \brief Copy voxel data from here to host
     *
     *  \param data   pointer to destination memory
     *  \param npitch step size in bytes of destination memory
     *  \return Returns \a cudaError_t (CUDA error code)
     */
    __host__ inline
    cudaError_t copyTo(fusionvoxel<_histBins> * data, size_t npitch)
    {
        if (memT == Device) return Device2HostCopy(data, npitch, stg.ptr, stg.pitch, stg.width, stg.height, stg.depth);
#if CUDA_VERSION_MAJOR >= 6
        if (memT == MemoryKind::Managed) return Device2HostCopy(data, npitch, stg.ptr, stg.pitch, stg.width, stg.height, stg.depth);
#endif // CUDA_VERSION_MAJOR >= 6
        return Host2HostCopy(data, npitch, stg.ptr, stg.pitch, stg.width, stg.height, stg.depth);
    }

    /**
     *  \brief Copy \a this to device and return pointer to it
     *
     *  \return Pointer to fusionData class on device
     *
     *  \details If data is already on the device, no copying of the data is taking place
     */
    __host__ inline
    fusionData<_histBins, Device> * toDevice()
    {
        size_t p = stg.pitch, s = stg.spitch;
        bool owned = stg.own;

        // Copy the elements to the device.
        fusionvoxel<_histBins> * voxels, * voxelsthis = stg.ptr;
        if (memT != Device) {
            size_t pitch, spitch;
            MemoryManagement<fusionvoxel<_histBins>, Device>::Malloc(voxels, stg.width, stg.height, stg.depth, pitch, spitch);
            Host2DeviceCopy(voxels, pitch, stg.ptr, stg.pitch, stg.width, stg.height, stg.depth);
            stg.pitch = pitch;
            stg.spitch = spitch;
        }
        else {
            voxels = stg.ptr;
            stg.own = false;
        }

        // Copy this to the device.
        stg.ptr = voxels;
        fusionData<_histBins, Device> * deviceArray;
        cudaMalloc((void **)&deviceArray, sizeof(fusionData<_histBins, memT>));
        cudaMemcpy((void *)deviceArray, this, sizeof(fusionData<_histBins, memT>),
                   cudaMemcpyHostToDevice);

        stg.pitch = p;
        stg.spitch = s;
        stg.ptr = voxelsthis;
        stg.own = owned;

        return deviceArray;
    }

    /**
     *  \brief Get all data associated with this fusionData
     *
     *  \return Struct with this fusionData members
     */
    __host__ __device__ inline
    fusionDataSettings<_histBins> exportSettings()
    {
        return stg;
    }

    /**
     *  \brief Get all data associated with this fusionData
     *
     *  \return Struct with this fusionData members
     */
    __host__ __device__ inline
    const fusionDataSettings<_histBins> exportSettings() const
    {
        return stg;
    }

    /**
     *  \brief Set this fusionData members to those in \p f.
     *
     *  \param f    struct with fusionData members
     *  \return No return value
     *
     *  \details No deep copy is done on voxel data, pointer to data is only given the address of
     *  voxel data in \p f. If \p this had allocated data which it controlled, it is deallocated.
     */
    __host__ __device__ inline
    void importSettings(fusionDataSettings<_histBins> & f)
    {
        if (stg.own) CleanUp(stg.ptr);
        stg = f;
    }

protected:

    /** \brief Structure with all the reuired data stored.
     * Required to ensure correct copying to device and back.*/
    fusionDataSettings<_histBins> stg;

    /**
     *  \brief Calculate and set histogram bin parameters
     *
     *  \return No return value
     */
    __host__ __device__ inline
    void binParams()
    {
        stg.bin[0] = -1.f;
        stg.bin[_histBins - 1] = 1.f;
        if (_histBins > 3){
            for (unsigned char i = 1; i < _histBins - 1; i++) stg.bin[i] = 2.f * float(i - 1) / float(_histBins - 3) - 1.f;
            stg.binstep = 2 / float(bins() - 3);
        }
        else {
            if (_histBins == 3) stg.bin[1] = 0.f;
            stg.binstep = 0.f;
        }
    }
};

/** \brief Convenience typedef for fusionData with 2 bins and stored on device */
typedef fusionData<2> dfusionData2;
/** \brief Convenience typedef for fusionData with 3 bins and stored on device */
typedef fusionData<3> dfusionData3;
/** \brief Convenience typedef for fusionData with 4 bins and stored on device */
typedef fusionData<4> dfusionData4;
/** \brief Convenience typedef for fusionData with 5 bins and stored on device */
typedef fusionData<5> dfusionData5;
/** \brief Convenience typedef for fusionData with 6 bins and stored on device */
typedef fusionData<6> dfusionData6;
/** \brief Convenience typedef for fusionData with 7 bins and stored on device */
typedef fusionData<7> dfusionData7;
/** \brief Convenience typedef for fusionData with 8 bins and stored on device */
typedef fusionData<8> dfusionData8;
/** \brief Convenience typedef for fusionData with 9 bins and stored on device */
typedef fusionData<9> dfusionData9;
/** \brief Convenience typedef for fusionData with 10 bins and stored on device */
typedef fusionData<10> dfusionData10;

/** \brief Convenience typedef for fusionData with 2 bins and stored on host */
typedef fusionData<2, Host> fusionData2;
/** \brief Convenience typedef for fusionData with 3 bins and stored on host */
typedef fusionData<3, Host> fusionData3;
/** \brief Convenience typedef for fusionData with 4 bins and stored on host */
typedef fusionData<4, Host> fusionData4;
/** \brief Convenience typedef for fusionData with 5 bins and stored on host */
typedef fusionData<5, Host> fusionData5;
/** \brief Convenience typedef for fusionData with 6 bins and stored on host */
typedef fusionData<6, Host> fusionData6;
/** \brief Convenience typedef for fusionData with 7 bins and stored on host */
typedef fusionData<7, Host> fusionData7;
/** \brief Convenience typedef for fusionData with 8 bins and stored on host */
typedef fusionData<8, Host> fusionData8;
/** \brief Convenience typedef for fusionData with 9 bins and stored on host */
typedef fusionData<9, Host> fusionData9;
/** \brief Convenience typedef for fusionData with 10 bins and stored on host */
typedef fusionData<10, Host> fusionData10;

/** @} */ // group fusion

#endif // FUSION_H
