#ifndef PLANESWEEP_H
#define PLANESWEEP_H

#define DEFAULT_Z_NEAR 0.1f
#define DEFAULT_Z_FAR 1.0f
#define DEFAULT_NUMBER_OF_PLANES 200
#define DEFAULT_NUMBER_OF_IMAGES 4
#define DEFAULT_WINDOW_SIZE 5
#define DEFAULT_STD_THRESHOLD 0.0001f
#define DEFAULT_NCC_THRESHOLD 0.5f
#define NO_DEPTH -1
#define NO_CUDA_DEVICE -1
#define MAX_THREADS_PER_BLOCK 512
#define MAX_PLANESWEEP_THREADS 1

#define DEFAULT_TVL1_ITERATIONS 100
#define DEFAULT_TVL1_LAMBDA .3
#define DEFAULT_TVL1_TAU 0.02
#define DEFAULT_TVL1_SIGMA 6.f
#define DEFAULT_TVL1_THETA 1.f
#define DEFAULT_TVL1_BETA 0.f
#define DEFAULT_TVL1_GAMMA 1.f

#define DEFAULT_BLOCK_XDIM 32

#define DEFAULT_TGV_LAMBDA 0.5
#define DEFAULT_TGV_ALPHA0 2.0
#define DEFAULT_TGV_ALPHA1 1.5
#define DEFAULT_TGV_NITERS 30
#define DEFAULT_TGV_NWARPS 15
#define DEFAULT_TGV_SIGMA 1.f
#define DEFAULT_TGV_TAU 0.02
#define DEFAULT_TGV_BETA 0.f
#define DEFAULT_TGV_GAMMA 1.f

#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <string.h>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <mutex>

namespace ublas = boost::numeric::ublas;
typedef unsigned char uchar;

class PlaneSweep
{
public:
    // Image structures:
    // data - image pixel data
    // pitch - size in bytes between consecutive rows
    // channels - number of channels (gray - 1, RGB - 3, etc.)
    template <typename T>
    class camImage {
    public:
        T * data;
        unsigned int pitch;
        uchar channels;
        unsigned int width;
        unsigned int height;
        ublas::matrix<double> R;
        ublas::matrix<double> t;

        camImage() :
			R(ublas::matrix<double>(3,3)),
            t(ublas::matrix<double>(3,1)),
            data(nullptr),
            allocated(false)
		{
        }

        void setSize(unsigned int w, unsigned int h, bool allocate_data = true){
			width = w;
			height = h;
            pitch = w * sizeof(T);
			delete[] data;
            if (allocate_data) data = new T[w * h];
            allocated = allocate_data;
		}

		void CopyFrom(const T * d, unsigned int elements){
			delete[] data;
			data = new T[elements];
			std::copy(d, d + elements, data);
			allocated = true;
		}

		void CopyFrom(const T *pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
		{
			delete[] data;
			data = new T[nWidth * nHeight];
			pitch = nWidth * sizeof(T);
            width = nWidth;
            height = nHeight;

			for (size_t iLine = 0; iLine < nHeight; ++iLine)
			{
				// copy one line worth of data
				std::copy_n(pSrc, nWidth, data);
				// move data pointers to next line
				data += pitch / sizeof(T);
				pSrc += nSrcPitch / sizeof(T);
			}

			// return pointer to beginning
			data -= pitch * height / sizeof(T);

			allocated = true;
		}

        camImage<T>& operator=(const camImage<T>& i){ 
            if (this == &i) return *this;

			CopyFrom(i.data, i.pitch, i.width, i.height);
            channels = i.channels;
            R = i.R;
            t = i.t;
            return *this;
        }

        ~camImage(){
			if (allocated) {
				delete[] data;
				data = nullptr;
			}
        }

    private:
        bool allocated;
    };

    // Reference and sensor views stored in float format
    camImage<float> HostRef;
    std::vector<camImage<float>> HostSrc;

    // same views stored in uchar format
    camImage<uchar> HostRef8u;
    std::vector<camImage<uchar>> HostSrc8u;

    PlaneSweep();
    PlaneSweep(int argc, char **argv);
    ~PlaneSweep ();

    bool RunAlgorithm(int argc, char **argv);
    bool Denoise(unsigned int niter, double lambda);
    bool CudaDenoise(int argc, char **argv, const unsigned int niters = DEFAULT_TVL1_ITERATIONS, const double lambda = DEFAULT_TVL1_LAMBDA,
                     const double tau = DEFAULT_TVL1_TAU, const double sigma = DEFAULT_TVL1_SIGMA, const double theta = DEFAULT_TVL1_THETA,
                     const double beta = DEFAULT_TVL1_BETA, const double gamma = DEFAULT_TVL1_GAMMA);
    bool TGV(int argc, char **argv, const unsigned int niters = DEFAULT_TGV_NITERS, const unsigned int warps = DEFAULT_TGV_NWARPS,
             const double lambda = DEFAULT_TGV_LAMBDA, const double alpha0 = DEFAULT_TGV_ALPHA0, const double alpha1 = DEFAULT_TGV_ALPHA1,
             const double tau = DEFAULT_TGV_TAU, const double sigma = DEFAULT_TGV_SIGMA,
             const double beta = DEFAULT_TGV_BETA, const double gamma = DEFAULT_TGV_GAMMA);
    void RelativeMatrices(ublas::matrix<double> & Rrel, ublas::matrix<double> & trel, const ublas::matrix<double> & Rref,
                          const ublas::matrix<double> & tref, const ublas::matrix<double> & Rsrc, const ublas::matrix<double> & tsrc);

    // Setters:
    void setAlternativeRelativeMatrixMethod(bool method) { alternativemethod = method; }
    void setK(double Km[][3]){ arrayToMatrix(Km, K); invertK(); }
    void setK(ublas::matrix<double> & Km){ K = Km; invertK(); }
    void setZ(float n, float f){ znear = n; zfar = f; }
    void setZnear(float n){ znear = n; }
    void setZfar(float f){ zfar = f; }
    void setNumberofPlanes(unsigned int n){ numberplanes = n; }
    void setNumberofImages(unsigned int n){ numberimages = n; }
    void setWindowSize(unsigned int sz){ if (sz % 2 == 0) std::cout << "Window size must be an odd number"; else winsize = sz; }
    void setSTDthreshold(float th){ stdthresh = th; }
    void setNCCthreshold(float th){ nccthresh = th; }
    void setThreadsPerBlock(dim3 tpb){ threads = tpb; }
    void setBlockXdim(int & threadsx){ if (threadsx * threads.y > maxThreadsPerBlock) threadsx = maxThreadsPerBlock / threads.y;
                                       threads.x = threadsx;}
    void setBlockYdim(int & threadsy){ if (threadsy * threads.x > maxThreadsPerBlock) threadsy = maxThreadsPerBlock / threads.x;
                                       threads.y = threadsy;}

    // Getters:
    bool getAlternativeRelativeMatrixMethod(){ return alternativemethod; }
    void getK(double k[][3]){ matrixToArray(k, K); }
    void getInverseK(double k[][3]){ matrixToArray(k, invK); }
    camImage<float> * getDepthmap(){ return &depthmap; }
    camImage<float> * getDepthmapDenoised(){ return &depthmapdenoised; }
    camImage<uchar> * getDepthmap8u(){ return &depthmap8u; }
    camImage<uchar> * getDepthmap8uDenoised(){ return &depthmap8udenoised; }
    camImage<float> * getDepthmapTGV(){ return &depthmapTGV; }
    camImage<uchar> * getDepthmap8uTGV(){ return &depthmap8uTGV; }
    void get3Dcoordinates(camImage<float> * &x, camImage<float> * &y, camImage<float> * &z){ x = &coord_x; y = &coord_y; z = &coord_z; }
    float getZnear(){ return znear; }
    float getZfar(){ return zfar; }
    unsigned int getNumberofPlanes(){ return numberplanes; }
    unsigned int getNumberofImages(){ return numberimages; }
    unsigned int getWindowSize(){ return winsize; }
    float getSTDthreshold(){ return stdthresh; }
    float getNCCthreshold(){ return nccthresh; }
    unsigned int getMaxThreadsPerBlock(){ return maxThreadsPerBlock; }
    dim3 getThreadsPerBlock(){ return threads; }

    // Images for planesweep:
    void setRreference(double R[][3]){ arrayToMatrix(R, HostRef.R); }
    void setRsource(unsigned int number, double R[][3]){ if (HostSrc.size() < number - 1) std::cout << "Not enough source images\n";
                                                        else arrayToMatrix(R, HostSrc[number].R); }
    void setTreference(double t[]){ TmatrixToArray(t, HostRef.t); }
    void setTsource(unsigned int number, double t[3]){ if (HostSrc.size() < number - 1) std::cout << "Not enough source images\n";
                                                        else TarrayToMatrix(t, HostSrc[number].t); }
    void setCreference(double C[][4]){ CtoRT(C, HostRef.R, HostRef.t); }
    void setCsource(unsigned int number, double C[][4]){ if (HostSrc.size() < number - 1) std::cout << "Not enough source images\n";
                                                        else CtoRT(C, HostSrc[number].R, HostSrc[number].t); }
    void Convert8uTo32f(int argc, char **argv);

    // array-matrix conversions for R, t and C
    void arrayToMatrix(double A[][3], ublas::matrix<double> &B){ B.resize(3,3, false); for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) B(i,j) = A[i][j]; }
    void matrixToArray(double A[][3], ublas::matrix<double> &B){ for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) A[i][j] = B(i,j);}
    void TarrayToMatrix(double T[], ublas::matrix<double> &t){ t.resize(3,1, false); t(0,0) = T[0]; t(1,0) = T[1]; t(2,0) = T[2]; }
    void TmatrixToArray(double T[], ublas::matrix<double> &t){ T[0] = t(0,0); T[1] = t(1,0); T[2] = t(2,0); }
    void CtoRT(double C[][4], ublas::matrix<double> &R, ublas::matrix<double> &t);
    void CmatrixToRT(ublas::matrix<double> &C, ublas::matrix<double> &R, ublas::matrix<double> &t);


protected:
    // K and inverse K matrices for camera
    ublas::matrix<double> K;
    ublas::matrix<double> invK;

    // vector normal to plane (0, 0, 1)T
    ublas::matrix<double> n;

    // stored depthmaps
    camImage<float> depthmap;
    camImage<float> depthmapdenoised;
    camImage<uchar> depthmap8u;
    camImage<uchar> depthmap8udenoised;
    camImage<float> depthmapTGV;
    camImage<uchar> depthmap8uTGV;

    camImage<float> coord_x, coord_y, coord_z;

    // plane sweep parameters
    float znear = DEFAULT_Z_NEAR;
    float zfar = DEFAULT_Z_FAR;

    unsigned int numberplanes = DEFAULT_NUMBER_OF_PLANES;
    unsigned int numberimages = DEFAULT_NUMBER_OF_IMAGES;
    unsigned int availableimages = 0;
    bool enoughimages = false;
    unsigned int winsize = DEFAULT_WINDOW_SIZE;

    float stdthresh = DEFAULT_STD_THRESHOLD;
    float nccthresh = DEFAULT_NCC_THRESHOLD;

    int maxThreadsPerBlock = MAX_THREADS_PER_BLOCK;
    int maxPlanesweepThreads = MAX_PLANESWEEP_THREADS;
    dim3 blocks, threads;

    bool depthavailable = false;
    bool alternativemethod = false;

    // inverse matrix function
    template<class T>
    bool InvertMatrix (const ublas::matrix<T>& input, ublas::matrix<T>& inverse);

    void ConvertDepthtoUChar(const camImage<float> &input, camImage<uchar> &output);

    // Single planesweep thread operating on single source view (all pointers point to memory on the GPU):
    void PlaneSweepThread(float * globDepth, float * globN, const float * Ref, const float * Refmean, const float * Refstd,
                          const unsigned int &index);

private:
    void invertK();

    // CUDA initialization functions
    int cudaDevInit(int argc, const char **argv);
    bool printfNPPinfo();

};

#endif // PLANESWEEP_H
