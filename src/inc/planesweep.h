/**
 *  \file planesweep.h
 *  \brief Header file containing PlaneSweep class implementation
 */
#ifndef PLANESWEEP_H
#define PLANESWEEP_H

#include "defines.h"
#include "structs.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>

typedef unsigned char uchar;

/** \addtogroup planesweep
* @{
*/

/**
*  \brief Class that implements depthmap generation methods using planesweep, TVL1 denoising
* and TGV Multiview Stereo algorithms
*
*  \details Kernel block parameters of GPU functions can be controlled by \a setThreadsPerBlock,
* \a setBlockXdim and \a setBlockYdim functions.
*/
class PlaneSweep
{
public:
    /** \brief Simple class to hold image data and camera position matrices on host */
    template <typename T>
    struct camImage {
    public:
        T * data;
        unsigned int pitch;
        uchar channels;
        unsigned int width;
        unsigned int height;
        /** \brief Rotation matrix */
        Matrix3D R;
        /** \brief Translation vector */
        Vector3D t;

        /** \brief Default constructor */
        camImage() :
            data(nullptr),
            allocated(false)
        {}

        /**
         *  \brief Resize image
         *
         *  \param w             width of data
         *  \param h             height of data
         *  \param allocate_data does data need to be allocated?
         *
         *  \details <b>Deletes any previous data</b>
         */
        void setSize(unsigned int w, unsigned int h, bool allocate_data = true){
            width = w;
            height = h;
            pitch = w * sizeof(T);
            delete[] data;
            if (allocate_data) data = new T[w * h];
            allocated = allocate_data;
        }

        /**
        *  \brief Perfom deep copy of \a data
        *
        *  \param d        pointer to data to be copied
        *  \param elements number of elements to copy
        *
        *  \details Any previous data is deleted and new memory allocated
        */
        void CopyFrom(const T * d, unsigned int elements){
            delete[] data;
            data = new T[elements];
            std::copy(d, d + elements, data);
            allocated = true;
        }

        /**
        *  \brief Perform deep copy of \a pSrc
        *
        *  \param pSrc      pointer to data to be copied
        *  \param nSrcPitch step size in bytes of source data
        *  \param nWidth    width of data
        *  \param nHeight   height of data
        *
        *  \details Any previous data is deleted and new memory allocated
        */
        void CopyFrom(const T *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
        {
            delete[] data;
            data = new T[nWidth * nHeight];
            T * ptr = data;
            pitch = nWidth * sizeof(T);
            width = nWidth;
            height = nHeight;

            for (size_t iLine = 0; iLine < nHeight; ++iLine)
            {
                // copy one line worth of data
                std::copy_n(pSrc, nWidth, data);
                // move data pointers to next line
                data = (T *)((uchar *)data + pitch);
                pSrc = (T *)((uchar *)pSrc + nSrcPitch);
            }

            // return pointer to beginning
            data = ptr;

            allocated = true;
        }

        /**
        *  \brief Copy operator
        *
        *  \details Performs deep copy of image data
        */
        camImage<T>& operator=(const camImage<T>& i){
            if (this == &i) return *this;

            CopyFrom(i.data, i.pitch, i.width, i.height);
            channels = i.channels;
            R = i.R;
            t = i.t;
            return *this;
        }

        /** \brief Default destructor.
        * If data was allocated by this camImage, it is deallocated. */
        ~camImage(){
            if (allocated) {
                delete[] data;
                data = nullptr;
            }
        }

    private:
        bool allocated;
    };

    /** \brief Reference image in float format */
    camImage<float> HostRef;

    /** \brief Source images in float format */
    std::vector<camImage<float>> HostSrc;

    /** \brief Reference image in unsigned char format */
    camImage<uchar> HostRef8u;

    /** \brief Source images in unsigned char format */
    std::vector<camImage<uchar>> HostSrc8u;

    /** \brief Default constructor */
    PlaneSweep();

    /**
    *  \brief Constructor overload
    *
    *  \param argc number of command line arguments
    *  \param argv pointers to command line argument strings
    */
    PlaneSweep(int argc, char **argv);

    /** \brief Default destructor */
    ~PlaneSweep ();

    /**
    *  \brief Planesweep algorithm
    *
    *  \param argc number of command line arguments
    *  \param argv pointers to command line argument strings
    *  \return Success/failure of the algorithm
    *
    *  \details Settings are changed \a set functions. Command line arguments
    * can be used to changed GPU used for the algorithm. Depthmaps can be retrieved by calling
    * \a getDepthmap() and \a getDepthmap8u() functions.
    */
    bool RunAlgorithm(int argc, char **argv);

    /**
    *  \brief \a OpenCV TVL1 denoising on CPU
    *
    *  \param niter  number of denoising iterations
    *  \param lambda TVL1 parameter \f$\lambda\f$
    *  \return Success/failure of the function
    *
    *  \details No other parameters are needed.
    * Denoising is performed on depthmap obtained by \a RunAlgorithm().
    * Depthmaps can be retrieved by calling
    * \a getDepthmapDenoised() and \a getDepthmap8uDenoised() functions.
    * * If there is no depthmap, function automatically returns false.
    * * If \a OpenCV library was not found, function returns false.
    */
    bool Denoise(unsigned int niter, double lambda);

    /**
    *  \brief TVL1 denoising on GPU
    *
    *  \param argc   number of command line arguments
    *  \param argv   pointers to command line argument strings
    *  \param niters number of denoising iterations
    *  \param lambda TVL1 parameter \f$\lambda\f$
    *  \param tau    TVL1 parameter \f$\tau\f$
    *  \param sigma  TVL1 parameter \f$\sigma\f$
    *  \param theta  TVL1 parameter \f$\theta\f$
    *  \param beta   anisotropic diffusion tensor parameter \f$\beta\f$
    *  \param gamma  anisotropic diffusion tensor parameter \f$\gamma\f$
    *  \return Success/failure of the function.
    *
    *  \details No more parameters are needed.
    * Denoising is performed on depthmap obtained by \a RunAlgorithm().
    * Depthmaps can be retrieved by calling
    * \a getDepthmapDenoised() and \a getDepthmap8uDenoised() functions.
    * * If there is no depthmap, function automatically returns false.
    */
    bool CudaDenoise(int argc, char **argv, const unsigned int niters = DEFAULT_TVL1_ITERATIONS, const double lambda = DEFAULT_TVL1_LAMBDA,
                     const double tau = DEFAULT_TVL1_TAU, const double sigma = DEFAULT_TVL1_SIGMA, const double theta = DEFAULT_TVL1_THETA,
                     const double beta = DEFAULT_TVL1_BETA, const double gamma = DEFAULT_TVL1_GAMMA);
    
    /**
    *  \brief TGV Multiview Stereo
    *
    *  \param argc   number of command line arguments
    *  \param argv   pointers to command line argument strings
    *  \param niters number of iterations
    *  \param warps  number of algorithm reinitializations with last depthmap values
    *  \param lambda TGV parameter \f$\lambda\f$
    *  \param alpha0 TGV parameter \f$\alpha_0\f$
    *  \param alpha1 TGV parameter \f$\alpha_1\f$
    *  \param tau    TGV parameter \f$\tau\f$
    *  \param sigma  TGV parameter \f$\sigma\f$
    *  \param beta   anisotropic diffusion tensor parameter \f$\beta\f$
    *  \param gamma  anisotropic diffusion tensor parameter \f$\gamma\f$
    *  \return Success/failure of the function.
    *
    *  \details Requires camera calibration matrix \f$K\f$ to be set with \a setK.
    * Depthmaps can be retrieved by calling
    * \a getDepthmapTGV() and \a getDepthmap8uTGV() functions.
    *
    * <b>UNOPTIMIZED</b>
    */
    bool TGV(int argc, char **argv, const unsigned int niters = DEFAULT_TGV_NITERS, const unsigned int warps = DEFAULT_TGV_NWARPS,
             const double lambda = DEFAULT_TGV_LAMBDA, const double alpha0 = DEFAULT_TGV_ALPHA0, const double alpha1 = DEFAULT_TGV_ALPHA1,
             const double tau = DEFAULT_TGV_TAU, const double sigma = DEFAULT_TGV_SIGMA,
             const double beta = DEFAULT_TGV_BETA, const double gamma = DEFAULT_TGV_GAMMA);

    /** \brief \b WIP Should increase accuracy of planesweep depthmap by using sparse accurate depthmap \p depth
     * Works well with anisotropic diffusion tensor and upto about x16 upscaling of sparse depthmap. */
    bool TGVdenoiseFromSparse(int argc, char **argv, const camImage<float> &depth, const unsigned int niters,
                              const double alpha0, const double alpha1, const double tau, const double sigma, const double theta,
                              const double beta, const double gamma);
    
    /**
    *  \brief Calculate relative rotation and translation from reference to source views
    *
    *  \param Rrel relative rotation matrix returned by reference, size will be (3,3)
    *  \param trel relative translation vector returned by reference, size will be (3,1)
    *  \param Rref reference view rotation matrix of size (3,3)
    *  \param tref reference view translation vector of size (3,1)
    *  \param Rsrc source view rotation matrix of size (3,3)
    *  \param tsrc source view translation vector of size (3,1)
    *
    *  \details Separate function is needed because some methods differ.
    * Control the method with \a setAlternativeRelativeMatrixMethod()
    */
    void RelativeMatrices(Matrix3D & Rrel, Vector3D & trel, const Matrix3D & Rref,
                          const Vector3D & tref, const Matrix3D & Rsrc, const Vector3D & tsrc) const;

    // Setters:
    /**
    *  \brief Control relative matrix calculation method
    *
    *  \param method relative matrix calculation method
    *
    *  \details Currently there are only 2 different methods, hence \a bool variable.
    * If there are more than 2 methods, this will change to \a enum.
    *
    * \a method = \a false - use for real images from source directory
    * \a method = \a true - use for living room images from http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
    */
    void setAlternativeRelativeMatrixMethod(bool method) { alternativemethod = method; }

    /**
    *  \brief Set camera calibration matrix overload
    *
    *  \param Km camera calibration matrix values
    *
    *  \details Must be set before using \a RunAlgorithm() or \a TGV
    */
    void setK(const Matrix3D & Km){ K = Km; invK = Km.inv(); }

    /**
    *  \brief Set planesweep near and far plane depths
    *
    *  \param n near plane depth
    *  \param f far plane depth
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setZ(float n, float f){ znear = n; zfar = f; }

    /**
    *  \brief Set planesweep near plane depth
    *
    *  \param n near plane depth
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setZnear(float n){ znear = n; }

    /**
    *  \brief Set planesweep far plane depth
    *
    *  \param f far plane depth
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setZfar(float f){ zfar = f; }

    /**
    *  \brief Set planesweep number of planes on single source iamge
    *
    *  \param n number of planes
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setNumberofPlanes(unsigned int n){ numberplanes = n; }

    /**
    *  \brief Set planesweep number of images to run algorithm on
    *
    *  \param n number of images
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setNumberofImages(unsigned int n){ numberimages = n; }

    /**
    *  \brief Set planesweep NCC window size
    *
    *  \param sz length of window side
    *
    *  \details Must be set before using \a RunAlgorithm()
    */
    void setWindowSize(unsigned int sz){ if (sz % 2 == 0) std::cout << "Window size must be an odd number"; else winsize = sz; }

    /**
    *  \brief Set planesweep STD threshold
    *
    *  \param th STD threshold
    *
    *  \details Must be set before using \a RunAlgorithm(). STD values below
    * \a th cause \a NCC for that pixel to be set to 0.
    */
    void setSTDthreshold(float th){ stdthresh = th; }

    /**
    *  \brief Set planesweep NCC threshold
    *
    *  \param th NCC threshold
    *
    *  \details Must be set before using \a RunAlgorithm(). Depthmap values with
    * \a NCC value below \a th are not used in depthmap averaging.
    */
    void setNCCthreshold(float th){ nccthresh = th; }

    /**
    *  \brief Set kernel block dimensions
    *
    *  \param tpb kernel block dimensions
    */
    void setThreadsPerBlock(dim3 tpb){ threads = tpb; }

    /**
    *  \brief Set kernel block x dimension
    *
    *  \param threadsx kernel block x dimension
    */
    void setBlockXdim(int & threadsx){ if (threadsx * threads.y > maxThreadsPerBlock) threadsx = maxThreadsPerBlock / threads.y;
        threads.x = threadsx;}

    /**
    *  \brief Set kernel block y dimension
    *
    *  \param threadsy kernel block y dimension
    */
    void setBlockYdim(int & threadsy){ if (threadsy * threads.x > maxThreadsPerBlock) threadsy = maxThreadsPerBlock / threads.x;
        threads.y = threadsy;}

    // Getters:
    /**
    *  \brief Get relative matrix calculation method
    *
    *  \return Relative matrix calculation method
    *
    *  \details Control method with \a setAlternativeRelativeMatrixMethod()
    */
    bool getAlternativeRelativeMatrixMethod() const { return alternativemethod; }

    /**
    *  \brief Get camera calibration matrix \f$K\f$
    *  \return Camera calibration matrix
    */
    Matrix3D getK() const { return K; }

    /**
    *  \brief Get camera calibration matrix \f$K^{-1}\f$
    *  \return Inverse camera calibration matrix
    */
    Matrix3D getInverseK() const { return invK; }

    /**
    *  \brief Get pointer to raw planesweep depthmap
    *
    *  \return pointer to raw planesweep depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a RunAlgorithm()
    */
    camImage<float> * getDepthmap(){ return &depthmap; }

    /**
    *  \brief Get pointer to denoised planesweep depthmap
    *
    *  \return pointer to denoised planesweep depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a RunAlgorithm() and \a CudaDenoise() or \a Denoise() after that
    */
    camImage<float> * getDepthmapDenoised(){ return &depthmapdenoised; }

    /**
    *  \brief Get pointer to denoised planesweep depthmap on device memory
    *
    *  \return pointer to denoised planesweep depthmap on the device
    *
    *  \details Data pointed to by the pointer is overwritten on each new \a CudaDenoise() call. Unsuccessful CudaDenoise() may lead
    * to this pointer being empty.
    */
    float * getDepthmapDenoisedPtr(){ return d_depthmap; }

    /**
    *  \brief Get pointer to raw normalized planesweep depthmap
    *
    *  \return pointer to raw normalized planesweep depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a RunAlgorithm() and scaled to range [0,255] from [znear,zfar]
    */
    camImage<uchar> * getDepthmap8u(){ return &depthmap8u; }

    /**
    *  \brief Get pointer to denoised normalized planesweep depthmap
    *
    *  \return pointer to denoised normalized planesweep depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a RunAlgorithm() and \a CudaDenoise() or \a Denoise() after that and scaled to range [0,255] from [znear,zfar]
    */
    camImage<uchar> * getDepthmap8uDenoised(){ return &depthmap8udenoised; }

    /**
    *  \brief Get pointer to TGV depthmap
    *
    *  \return pointer to TGV depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a TGV()
    */
    camImage<float> * getDepthmapTGV(){ return &depthmapTGV; }

    /**
    *  \brief Get pointer to normalized TGV depthmap
    *
    *  \return pointer to normalized TGV depthmap
    *
    *  \details Depthmap returned is the last one calculated by running \a TGV() and scaled to range [0,255] from [znear,zfar]
    */
    camImage<uchar> * getDepthmap8uTGV(){ return &depthmap8uTGV; }

    /**
    *  \brief Get 3D coordinates of each camera pixel
    *
    *  \param x pointer to x coordinate, returned by reference
    *  \param y pointer to y coordinate, returned by reference
    *  \param z pointer to z coordinate, returned by reference
    *
    *  \details Coordinates are computed at the time of the call.
    */
    void get3Dcoordinates(camImage<float> * &x, camImage<float> * &y, camImage<float> * &z);

    /**
    *  \brief Get currently set planesweep near plane depth
    *
    *  \return Planesweep near plane depth
    */
    float getZnear() const { return znear; }

    /**
    *  \brief Get currently set planesweep far plane depth
    *
    *  \return Planesweep far plane depth
    */
    float getZfar() const { return zfar; }

    /**
    *  \brief Get currently set planesweep number of planes
    *
    *  \return Planesweep number of planes
    */
    unsigned int getNumberofPlanes() const { return numberplanes; }

    /**
    *  \brief Get currently set planesweep number of source images
    *
    *  \return Planesweep number of source images
    */
    unsigned int getNumberofImages() const { return numberimages; }

    /**
    *  \brief Get currently set planesweep NCC window size
    *
    *  \return Planesweep NCC window side length
    */
    unsigned int getWindowSize() const { return winsize; }

    /**
    *  \brief Get currently set planesweep STD threshold
    *
    *  \return Planesweep STD threshold
    */
    float getSTDthreshold() const { return stdthresh; }

    /**
    *  \brief Get currently set planesweep NCC threshold
    *
    *  \return Planesweep NCC threshold
    */
    float getNCCthreshold() const { return nccthresh; }

    /**
    *  \brief Get currently selected GPU threads per block limit
    *
    *  \return Threads per block limit
    */
    unsigned int getMaxThreadsPerBlock() const { return maxThreadsPerBlock; }

    /**
    *  \brief Get currently set kernel block dimensions
    *
    *  \return Kernel block dimensions
    */
    dim3 getThreadsPerBlock() const { return threads; }

protected:
    // K and inverse K matrices for camera
    Matrix3D K;
    Matrix3D invK;

    // stored depthmaps
    camImage<float> depthmap;
    camImage<float> depthmapdenoised;
    camImage<uchar> depthmap8u;
    camImage<uchar> depthmap8udenoised;
    camImage<float> depthmapTGV;
    camImage<uchar> depthmap8uTGV;

    // pointer to depthmap on the device after TVL1 denoising
    float * d_depthmap;

    // stored coordinates
    camImage<float> coord_x, coord_y, coord_z;

    // plane sweep parameters
    float znear = DEFAULT_Z_NEAR;
    float zfar = DEFAULT_Z_FAR;
    unsigned int numberplanes = DEFAULT_NUMBER_OF_PLANES;
    unsigned int numberimages = DEFAULT_NUMBER_OF_IMAGES;
    unsigned int winsize = DEFAULT_WINDOW_SIZE;
    float stdthresh = DEFAULT_STD_THRESHOLD;
    float nccthresh = DEFAULT_NCC_THRESHOLD;

    // CUDA kernel parameters
    int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int maxPlanesweepThreads = MAX_PLANESWEEP_THREADS;
    dim3 blocks, threads;

    // PlaneSweep method flags
    bool depthavailable = false;
    bool alternativemethod = false;

    /**
    *  \brief Depthmap normalization function for easy representation as grayscale image
    *
    *  \param input  depthmap input
    *  \param output normalized depthmap output returned by reference
    *
    *  \details Depthmap is scaled to range [0,255] from [znear,zfar]
    */
    void ConvertDepthtoUChar(const camImage<float> &input, camImage<uchar> &output);

    /**
    *  \brief Single planesweep thread operating on single source view (all pointers point to memory on the GPU):
    *
    *  \param globDepth pointer to sum of depthmaps
    *  \param globN     pointer to depthmap summation count
    *  \param Ref       pointer to reference intensity image
    *  \param Refmean  pointer to reference windowed means image
    *  \param Refstd    pointer to reference windowed STD image
    *  \param index     index of source view image in \a std::vector
    *
    *  \details Multithreading does not increase performance
    */
    void PlaneSweepThread(float * globDepth, float * globN, const float * Ref, const float * Refmean, const float * Refstd,
                          const unsigned int &index);

private:

    // CUDA initialization functions
    int cudaDevInit(int argc, const char **argv);
    bool printfNPPinfo();

    /** \brief Cuda GPU reset function. Use this function to reallocate memory on the device when it
    * encountered an error and needed a reset */
    void cudaReset();

};

/** @} */ // group planesweep

#endif // PLANESWEEP_H
