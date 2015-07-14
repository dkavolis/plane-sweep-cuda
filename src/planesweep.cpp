#include <planesweep.h>
#include <iostream>
#include <chrono>

// Boost:
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

// CUDA: (header files contain definitions)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <npp.h>
#include <ImagesNPP.h>
#include <cuda_runtime.h>
#include <helper_string.h>
#include <kernels.cu.h>

// wrappers for NPP float image processing functions
// windowed functions take kernel as input in addition to windowsize to reduce time spent by allocating it on the GPU
// note that kernel length should be an odd number and all its elements equal to 1 / windowsize
// windowed mean functions use 2 1D filters to further reduce function time
void WindowedMean32f(npp::ImageNPP_32f_C1 & input, unsigned int windowsize, Npp32f *pKernel, npp::ImageNPP_32f_C1 & output);
void WindowedMeanSquares32f(npp::ImageNPP_32f_C1 & input, unsigned int windowsize, Npp32f* pKernel, npp::ImageNPP_32f_C1 & output);
void Positive32f(npp::ImageNPP_32f_C1 & input, npp::ImageNPP_32f_C1 & output); // if input .< 0, set input = 0
void GreaterEqual32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_8u_C1 & output); // A .>= B
void GreaterEqualC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_8u_C1 & output); // A >= C
void Product32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output); // A .* B
void ProductC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output); // A * C
void RDivide32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output); // A ./ B
void RDivideC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output); // A / C
void Sum32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output); // A .+ B
void Sum8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & B, npp::ImageNPP_8u_C1 & output);
void SumC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output); // A + C
void Difference32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output); // A .- B
void DifferenceC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output); // A - C
void SQRT32f(npp::ImageNPP_32f_C1 & input, npp::ImageNPP_32f_C1 & output); // square root of input pixels
void AND8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & B, npp::ImageNPP_8u_C1 & output);
void NOT8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & output);
void Conversion8u32f(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_32f_C1 & output);

template <typename T> // T models Any
struct static_cast_func
{
    template <typename T1> // T1 models type statically convertible to T
    T operator()(const T1& x) const { return static_cast<T>(x); }
};

int PlaneSweep::cudaDevInit(int argc, const char **argv)
{
    int Count;
    checkCudaErrors(cudaGetDeviceCount(&Count));

    if (Count == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        return NO_CUDA_DEVICE;
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProps.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;

    return dev;
}

bool PlaneSweep::printfNPPinfo()
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool Val = checkCudaCapabilities(1, 0);
    return Val;
}

PlaneSweep::PlaneSweep()
{
    K.resize(3,3);
    invK.resize(3,3);
    n.resize(3,1);

    // Set correct n elements
    n(0,0) = 0;
    n(1,0) = 0;
    n(2,0) = 1;
}

bool PlaneSweep::RunAlgorithm(int argc, char **argv)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    // Reset depthmap

    depthmap.setSize(HostRef.width, HostRef.height);
    depthmap.channels = 1;
    depthmap8u.setSize(HostRef.width, HostRef.height);

    printf("Starting plane sweep algorithm...\n\n");

    try
    {
        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaDeviceReset();
            return false;
        }

        if (printfNPPinfo() == false)
        {
            cudaDeviceReset();
            return false;
        }

        // Algorithm here:-------------------------------------

        // Move reference image to device memory
        NppiSize imSize={(int)HostRef.width, (int)HostRef.height};
        npp::ImageNPP_32f_C1 deviceRef(imSize.width, imSize.height);
        deviceRef.copyFrom(HostRef.data, HostRef.pitch);
        float * temp = new float [imSize.height * imSize.width];

        dim3 threads(32,32);
        dim3 blocks(ceil(imSize.width/(float)threads.x), ceil(imSize.height/(float)threads.y));

        // Create summation kernel on host memory
        Npp32f* hostKernel = new Npp32f [winsize];
        for (int i = 0; i < winsize; i++) {
            hostKernel[i] = 1.f / winsize;
        }

        // Transfer the kernel to device memory
        Npp32f* pKernel; //just a regular 1D array on the GPU
        NPP_CHECK_CUDA(cudaMalloc((void**)&pKernel, winsize * sizeof(Npp32f)));
        NPP_CHECK_CUDA(cudaMemcpy(pKernel, hostKernel, winsize * sizeof(Npp32f), cudaMemcpyHostToDevice));

        // Create images on the device to hold windowed mean and std for reference image + intermediate images
        // and calculate the images
        npp::ImageNPP_32f_C1 deviceRefmean(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 deviceRefstd(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devInter1(imSize.width, imSize.height); // intermediate image, will hold square of means in this computation
        WindowedMean32f(deviceRef, winsize, pKernel, deviceRefmean); // windowed mean
        WindowedMeanSquares32f(deviceRef, winsize, pKernel, devInter1); // mean of squares
        calculate_STD(deviceRefstd.data(), deviceRefmean.data(),
                      devInter1.data(), imSize.width, imSize.height, blocks, threads);

        // calculate depth step size:
        float dstep = (zfar - znear) / (numberplanes - 1);

        // Create images to hold depthmap values and number of times it exceeded NCC threshold
        npp::ImageNPP_32f_C1 devDepthmap(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devN(imSize.width, imSize.height);
        npp::ImageNPP_8u_C1 devN8u(imSize.width, imSize.height);

        // set count to 0
        set_value(devN.data(), 0.f, imSize.width, imSize.height, blocks, threads);

        // set depthmap values to 0
        set_value(devDepthmap.data(), 0.f, imSize.width, imSize.height, blocks, threads);

        // Create image to store current source view
        npp::ImageNPP_32f_C1 devSrc(imSize.width, imSize.height);

        // Create matrices to hold homography and relative rotation and transformation
        ublas::matrix<double> H(3,3), Rrel(3,3), trel(3,1);

        // Store and calculate inverse reference rotation matrix
        ublas::matrix<double> invR(3,3);
        InvertMatrix(HostRef.R, invR);

        // Create intermediate images to store current NCC, best NCC and current depthmap
        npp::ImageNPP_32f_C1 devNCC(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devbestNCC(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devDepth(imSize.width, imSize.height);

        // Create images to store x and y indexes after transformation
        npp::ImageNPP_32f_C1 devx(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devy(imSize.width, imSize.height);

        // Create image to hold pixel values after transformation
        npp::ImageNPP_32f_C1 devWarped(imSize.width, imSize.height);

        for (int i = 0; i < std::min(std::max((int)numberimages, 1), (int)HostSrc.size()); i++){

            // Copy source view to device
            devSrc.copyFrom(HostSrc[i].data, HostSrc[i].pitch);

            // Reset best ncc and depthmap
            set_value(devbestNCC.data(), 0.f, imSize.width, imSize.height, blocks, threads);
            set_value(devDepth.data(), 0.f, imSize.width, imSize.height, blocks, threads);

            // Calculate relative rotation and translation:
            Rrel = prod(HostSrc[i].R, invR);
            trel = HostSrc[i].t - prod(Rrel, HostRef.t);

            // For each depth calculate NCC and update depthmap as required
            for (float d = znear; d <= zfar; d += dstep){

                // Calculate homography:
                H = Rrel + prod(trel, trans(n)) / d;
                H = prod(K, H);
                H = prod(H, invK);
                H = H / H(2,2);

                // Calculate transformed pixel coordinates
                affine_transform_indexes(devx.data(), devy.data(),
                                         H(0,0), H(0,1), H(0,2),
                                         H(1,0), H(1,1), H(1,2),
                                         imSize.width, imSize.height, blocks, threads);

                // interpolate pixel values:
                bilinear_interpolation(devWarped.data(), devSrc.data(),
                                       devx.data(), devy.data(),
                                       devSrc.width(), devSrc.height(),
                                       devx.width(), devx.height(),
                                       blocks, threads);


                // We have no more use for devx and devy, we can use them to store intermediate results now
                // devx - will hold windowed mean of warped image
                // devy - will hold windowed std of warped image
                WindowedMean32f(devWarped, winsize, pKernel, devx); // windowed mean
                WindowedMeanSquares32f(devWarped, winsize, pKernel, devInter1); // mean of squares
                calculate_STD(devy.data(), devx.data(), devInter1.data(),
                              imSize.width, imSize.height, blocks, threads);

                // calculate NCC for each window which is given by
                // NCC = (mean of products - product of means) / product of standard deviations
                element_multiply(devInter1.data(), deviceRef.data(), devWarped.data(), imSize.width, imSize.height, blocks, threads);
                WindowedMean32f(devInter1, winsize, pKernel, devWarped); // windowed mean of products
                calcNCC(devNCC.data(), devWarped.data(),
                        deviceRefmean.data(), devx.data(),
                        deviceRefstd.data(), devy.data(),
                        stdthresh, stdthresh,
                        imSize.width, imSize.height,
                        blocks, threads);

                // only keep depth and bestncc values for which best ncc is greater than current
                // set other values to current ncc and depth
                update_arrays(devDepth.data(), devbestNCC.data(),
                              devNCC.data(), d, imSize.width, imSize.height,
                              blocks, threads);

            }

            // Sum depth results for later averaging
            sum_depthmap_NCC(devDepthmap.data(), devN.data(),
                             devDepth.data(), devbestNCC.data(),
                             nccthresh, imSize.width, imSize.height,
                             blocks, threads);
        }

        // Calculate averaged depthmap and copy it to host
        element_rdivide(devDepthmap.data(), devDepthmap.data(), devN.data(), imSize.width, imSize.height, blocks, threads);
        devDepthmap.copyTo(depthmap.data, depthmap.pitch);
        convert_float_to_uchar(devN8u.data(), devDepthmap.data(), znear, zfar, imSize.width, imSize.height, blocks, threads);
        devN8u.copyTo(depthmap8u.data, depthmap8u.pitch);
        printf("%d pitch, %d width, %d height, %f znear, %f zfar, %d imwidth, %d imheight\n", depthmap8u.pitch, depthmap8u.width,
               depthmap8u.height, znear, zfar, imSize.width, imSize.height);
        int y = 100;
        //devDepthmap.copyTo(temp, imSize.width*4);
        for (int i = 0; i < imSize.height; i++) printf("x = %d, y = %d, val uchar = %d, val float = %f\n",
                                                       y, i+1, depthmap8u.data[(i)*imSize.width + y-1], depthmap.data[(i)*imSize.width + y-1]);

        // Free up resources
        nppiFree(deviceRef.data());
        nppiFree(deviceRefmean.data());
        nppiFree(deviceRefstd.data());
        nppiFree(devInter1.data());
        nppiFree(devDepthmap.data());
        nppiFree(devN.data());
        nppiFree(devN8u.data());
        nppiFree(devSrc.data());
        nppiFree(devNCC.data());
        nppiFree(devbestNCC.data());
        nppiFree(devDepth.data());
        nppiFree(devx.data());
        nppiFree(devy.data());
        nppiFree(devWarped.data());
        checkCudaErrors(cudaFree(pKernel));
        delete[] hostKernel;
        delete[] temp;

        //-----------------------------------------------------

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the algorithm to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";

        cudaDeviceReset();
        return true;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaDeviceReset();
        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        cudaDeviceReset();
        return false;

    }
    return false;
}

void PlaneSweep::Convert8uTo32f(int argc, char **argv)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Starting conversion...\n\n");

    try
    {

        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaDeviceReset();
            return;
        }

        if (printfNPPinfo() == false)
        {
            cudaDeviceReset();
            return;
        }
        // Algorithm here:-------------------------------------

        int w = HostRef8u.width;
        int h = HostRef8u.height;

        // Create 32f and 8u device images
        npp::ImageNPP_8u_C1 im8u(w, h);
        npp::ImageNPP_32f_C1 im32f(w, h);

        // convert reference image
        HostRef.setSize(w,h);
        im8u.copyFrom(HostRef8u.data, HostRef8u.pitch);
        Conversion8u32f(im8u, im32f);
        im32f.copyTo(HostRef.data, HostRef.pitch);
        HostRef.R = HostRef8u.R;
        HostRef.t = HostRef8u.t;

        HostSrc.resize(HostSrc8u.size());

        // convert source views
        for (int i = 0; i < HostSrc8u.size(); i++){
            HostSrc[i].setSize(w, h);
            im8u.copyFrom(HostSrc8u[i].data, HostSrc8u[i].pitch);
            Conversion8u32f(im8u, im32f);
            im32f.copyTo(HostSrc[i].data, HostSrc[i].pitch);
            HostSrc[i].R = HostSrc8u[i].R;
            HostSrc[i].t = HostSrc8u[i].t;

        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the conversion to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";

        // Free up resources
        nppiFree(im8u.data());
        nppiFree(im32f.data());

        //-----------------------------------------------------

        cudaDeviceReset();
        return;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaDeviceReset();
        return;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        cudaDeviceReset();
        return;

    }
}

void PlaneSweep::invertK()
{
    double detK = 1.0;
    for (int i = 0; i < 3; i++) detK *= K(i,i);

    invK(0,0) = K(1,1) / detK;
    invK(0,1) = - K(0,1) / detK;
    invK(0,2) = (K(0,1)*K(1,2) - K(0,2)*K(1,1)) / detK;

    invK(1,0) = 0.0;
    invK(1,1) = K(0,0) / detK;
    invK(1,2) = - K(1,2) / K(1,1);

    invK(2,0) = 0.0;
    invK(2,1) = 0.0;
    invK(2,2) = 1.0;

}

/* Matrix inversion routine.
    Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */
template<class T>
bool PlaneSweep::InvertMatrix (const ublas::matrix<T>& input, ublas::matrix<T>& inverse)
{
    using namespace boost::numeric::ublas;
    typedef permutation_matrix<std::size_t> pmatrix;
    // create a working copy of the input
    matrix<T> A(input);
    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = lu_factorize(A,pm);
    if( res != 0 ) return false;

    // create identity matrix of "inverse"
    inverse.assign(ublas::identity_matrix<T>(A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(A, pm, inverse);

    return true;
}

void PlaneSweep::CtoRT(double C[][4], ublas::matrix<double> &R, ublas::matrix<double> &t)
{
    R.resize(3,3);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R(i,j) = C[i][j];
    t.resize(3,1);
    t(0,0) = C[0][3];
    t(1,0) = C[1][3];
    t(2,0) = C[2][3];
}

void PlaneSweep::CmatrixToRT(ublas::matrix<double> &C, ublas::matrix<double> &R, ublas::matrix<double> &t)
{
    R.resize(3,3);
    t.resize(3,1);

    t <<= C(0,3), C(1,3), C(2,3);
    R <<= C(0,0), C(0,1), C(0,2),
            C(1,0), C(1,1), C(1,2),
            C(2,0), C(2,1), C(2,2);
}

PlaneSweep::~PlaneSweep()
{
    cudaDeviceReset();
}

void WindowedMean32f(npp::ImageNPP_32f_C1 & input, unsigned int windowsize, Npp32f* pKernel, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSrcSize = {(int)input.width(), (int)input.height()};
    NppiSize oSizeROI = {(int)input.width(), (int)input.height()};
    NppiPoint oSrcOffset = {0, 0};
    NPP_CHECK_NPP(nppiFilterColumnBorder_32f_C1R(input.data(), input.pitch(),
                                                 oSrcSize, oSrcOffset,
                                                 output.data(), output.pitch(),
                                                 oSizeROI, pKernel, windowsize, windowsize / 2, NPP_BORDER_REPLICATE));

    NPP_CHECK_NPP(nppiFilterRowBorder_32f_C1R(output.data(), output.pitch(),
                                              oSrcSize, oSrcOffset,
                                              output.data(), output.pitch(),
                                              oSizeROI, pKernel, (Npp32s)windowsize, (Npp32s)windowsize / 2, NPP_BORDER_REPLICATE));
}

void WindowedMeanSquares32f(npp::ImageNPP_32f_C1 & input, unsigned int windowsize, Npp32f *pKernel, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSrcSize = {(int)input.width(), (int)input.height()};
    NppiPoint oSrcOffset = {0, 0};
    NPP_CHECK_NPP(nppiSqr_32f_C1R(input.data(), input.pitch(), output.data(), output.pitch(), oSrcSize));
    NPP_CHECK_NPP(nppiFilterColumnBorder_32f_C1R(output.data(), output.pitch(),
                                                 oSrcSize, oSrcOffset,
                                                 output.data(), output.pitch(),
                                                 oSrcSize, pKernel, (Npp32s)windowsize, (Npp32s)windowsize / 2, NPP_BORDER_REPLICATE));

    NPP_CHECK_NPP(nppiFilterRowBorder_32f_C1R(output.data(), output.pitch(),
                                              oSrcSize, oSrcOffset,
                                              output.data(), output.pitch(),
                                              oSrcSize, pKernel, (Npp32s)windowsize, (Npp32s)windowsize / 2, NPP_BORDER_REPLICATE));
}

void Positive32f(npp::ImageNPP_32f_C1 & input, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)input.width(), (int)input.height()};
    NPP_CHECK_NPP(nppiThreshold_32f_C1R(input.data(), input.pitch(), output.data(), output.pitch(), oSize, 0, NPP_CMP_LESS));
}

void GreaterEqual32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_8u_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiCompare_32f_C1R(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize, NPP_CMP_GREATER_EQ));
}

void GreaterEqualC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_8u_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiCompareC_32f_C1R(A.data(), A.pitch(), C, output.data(), output.pitch(), oSize, NPP_CMP_GREATER_EQ));
}

void Product32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiMul_32f_C1R(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize));
}

void ProductC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiMulC_32f_C1R(A.data(), A.pitch(), C, output.data(), output.pitch(), oSize));
}

void RDivide32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiDiv_32f_C1R(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize));
}

void RDivideC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiDivC_32f_C1R(A.data(), A.pitch(), C, output.data(), output.pitch(), oSize));
}

void Sum32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiAdd_32f_C1R(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize));
}

void Sum8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & B, npp::ImageNPP_8u_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiAdd_8u_C1RSfs(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize, 0));
}

void SumC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiAddC_32f_C1R(A.data(), A.pitch(), C, output.data(), output.pitch(), oSize));
}

void Difference32f(npp::ImageNPP_32f_C1 & A, npp::ImageNPP_32f_C1 & B, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiSub_32f_C1R(B.data(), B.pitch(), A.data(), A.pitch(), output.data(), output.pitch(), oSize));
}

void DifferenceC32f(npp::ImageNPP_32f_C1 & A, float C, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiSubC_32f_C1R(A.data(), A.pitch(), C, output.data(), output.pitch(), oSize));
}

void SQRT32f(npp::ImageNPP_32f_C1 & input, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)input.width(), (int)input.height()};
    NPP_CHECK_NPP(nppiSqrt_32f_C1R(input.data(), input.pitch(), output.data(), output.pitch(), oSize));
}

void AND8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & B, npp::ImageNPP_8u_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiAnd_8u_C1R(A.data(), A.pitch(), B.data(), B.pitch(), output.data(), output.pitch(), oSize));
}

void NOT8u(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_8u_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiNot_8u_C1R(A.data(), A.pitch(),output.data(), output.pitch(), oSize));
}

void Conversion8u32f(npp::ImageNPP_8u_C1 & A, npp::ImageNPP_32f_C1 & output)
{
    NppiSize oSize = {(int)A.width(), (int)A.height()};
    NPP_CHECK_NPP(nppiConvert_8u32f_C1R(A.data(), A.pitch(), output.data(), output.pitch(), oSize));
}
