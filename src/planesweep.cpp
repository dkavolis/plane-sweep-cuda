#include "planesweep.h"
#include <chrono>
#include <fstream>
#include <thread>
#include <mutex>

// OpenCV:
#ifdef OpenCV_FOUND
#   include <opencv2/photo.hpp>
#   include <opencv2/core/mat.hpp>
#endif

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
#include <helper_cuda.h>
#include <helper_structs.h>

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
//    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProps.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    maxThreadsPerBlock = deviceProps.maxThreadsPerBlock;
//    std::cerr << "Max pitch allowed = " << deviceProps.memPitch << std::endl;
//    std::cerr << "Max grid dimensions: x = " << deviceProps.maxGridSize[0] << ", y = " << deviceProps.maxGridSize[1] << ", z = " <<
//                 deviceProps.maxGridSize[2] << std::endl;
//    std::cerr << "Max block dimensions: x = " << deviceProps.maxThreadsDim[0] << ", y = " << deviceProps.maxThreadsDim[1] << ", z = " <<
//                 deviceProps.maxThreadsDim[2] << std::endl;
//    std::cerr << "Warp size = " << deviceProps.warpSize << std::endl << std::endl;

    return dev;
}

bool PlaneSweep::printfNPPinfo()
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

//    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

//    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
//    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool Val = checkCudaCapabilities(1, 0);
    return Val;
}

PlaneSweep::PlaneSweep() :
    threads(dim3 (DEFAULT_BLOCK_XDIM)),
    d_depthmap(0)
{
}

PlaneSweep::PlaneSweep(int argc, char **argv) :
    d_depthmap(0)
{
    cudaDevInit(argc, (const char **)argv);
    threads = dim3(DEFAULT_BLOCK_XDIM, maxThreadsPerBlock/DEFAULT_BLOCK_XDIM);
    cudaReset();
}

bool PlaneSweep::RunAlgorithm(int argc, char **argv)
{
    auto t1 = std::chrono::high_resolution_clock::now();

    // Reset depthmap

    depthmap.setSize(HostRef.width, HostRef.height);
    depthmap.channels = 1;

    printf("Starting plane sweep algorithm...\n\n");

    try
    {
        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaReset();
            return false;
        }

        if (printfNPPinfo() == false)
        {
            cudaReset();
            return false;
        }

        // Algorithm here:-------------------------------------

        // Move reference image to device memory
        NppiSize imSize={(int)HostRef.width, (int)HostRef.height};
        npp::ImageNPP_32f_C1 deviceRef(imSize.width, imSize.height);
        deviceRef.copyFrom(HostRef.data, HostRef.pitch);

        if (threads.x * threads.y == 0) threads = dim3(DEFAULT_BLOCK_XDIM, maxThreadsPerBlock/DEFAULT_BLOCK_XDIM);
        blocks = dim3(ceil(imSize.width/(float)threads.x), ceil(imSize.height/(float)threads.y));

        // Create images on the device to hold windowed mean and std for reference image + intermediate images
        // and calculate the images
        npp::ImageNPP_32f_C1 deviceRefmean(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 deviceRefstd(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devInter1(imSize.width, imSize.height); // intermediate image, will hold square of means in this computation
        windowed_mean_column(devInter1.data(), deviceRef.data(), winsize, false, imSize.width, imSize.height,
                             blocks, threads);
        windowed_mean_row(deviceRefmean.data(), devInter1.data(), winsize, false, imSize.width, imSize.height,
                          blocks, threads);

        windowed_mean_column(devInter1.data(), deviceRef.data(), winsize, true, imSize.width, imSize.height,
                             blocks, threads);
        windowed_mean_row(deviceRefstd.data(), devInter1.data(), winsize, false, imSize.width, imSize.height,
                          blocks, threads);

        calculate_STD(deviceRefstd.data(), deviceRefmean.data(),
                      deviceRefstd.data(), imSize.width, imSize.height, blocks, threads);

        // Create images to hold depthmap values and number of times it exceeded NCC threshold
        npp::ImageNPP_32f_C1 devDepthmap(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devN(imSize.width, imSize.height);

        int nimgs = std::min(std::max((int)numberimages, 1), (int)HostSrc.size());
        int nthreads = std::min(nimgs, maxPlanesweepThreads);
        std::vector<std::thread> thread(nthreads);

        for (int i = 0; i < nimgs; i++){
            if (nthreads > 1){
                int ithread = i % nthreads;

                if (i / nthreads > 0) {
                    if (thread[ithread].joinable()) thread[ithread].join();
                }

                thread[ithread] = std::thread(&PlaneSweep::PlaneSweepThread, this, devDepthmap.data(), devN.data(),
                                              deviceRef.data(), deviceRefmean.data(), deviceRefstd.data(), i);
            }
            else PlaneSweep::PlaneSweepThread(devDepthmap.data(), devN.data(), deviceRef.data(), deviceRefmean.data(), deviceRefstd.data(), i);
        }

        for (int i = 0; i < nthreads; i++) {
            if (thread[i].joinable()) thread[i].join();
        }

        // Calculate averaged depthmap
        element_rdivide(devDepthmap.data(), devDepthmap.data(), devN.data(), imSize.width, imSize.height, blocks, threads);
        set_QNAN_value(devDepthmap.data(), zfar, imSize.width, imSize.height, blocks, threads);

        // Check for kernel errors
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) std::cerr << cudaGetErrorString(err) << std::endl;
        NPP_CHECK_CUDA(cudaPeekAtLastError());

        // Copy depthmap to host
        devDepthmap.copyTo(depthmap.data, depthmap.pitch);
        ConvertDepthtoUChar(depthmap, depthmap8u);
        depthavailable = true;

        // Free up resources
        nppiFree(deviceRef.data());
        nppiFree(deviceRefmean.data());
        nppiFree(deviceRefstd.data());
        nppiFree(devInter1.data());
        nppiFree(devDepthmap.data());
        nppiFree(devN.data());

        //-----------------------------------------------------

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the algorithm to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";
        std::cout.flush();

        return true;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaReset();
        return false;
    }
    catch(const std::system_error& e)
    {
        std::cerr << "Caught system error with code " << e.code()
                  << " meaning " << e.what() << '\n';

        return false;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Program error! The following std exception occurred: \n";
        std::cerr << e.what() << std::endl;
        std::cerr << "Aborting." << std::endl;

        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        return false;
    }
    return false;
}

void PlaneSweep::PlaneSweepThread(float *globDepth, float *globN, const float *Ref, const float *Refmean, const float *Refstd,
                                  const unsigned int &index)
{
    static std::mutex mutex;
    try {
        NppiSize imSize={(int)HostRef.width, (int)HostRef.height};

        // calculate depth step size:
        float dstep = (zfar - znear) / (numberplanes - 1);

        // Create image to store current source view
        npp::ImageNPP_32f_C1 devSrc(imSize.width, imSize.height);

        // Create matrices to hold homography and relative rotation and transformation
        Matrix3D H, Rrel, tr;
        Vector3D trel;

        // Create intermediate images to store current NCC, best NCC and current depthmap
        npp::ImageNPP_32f_C1 devNCC(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devbestNCC(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devDepth(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devInter1(imSize.width, imSize.height);

        // Create images to store x and y indexes after transformation
        npp::ImageNPP_32f_C1 devx(imSize.width, imSize.height);
        npp::ImageNPP_32f_C1 devy(imSize.width, imSize.height);

        // Create image to hold pixel values after transformation
        npp::ImageNPP_32f_C1 devWarped(imSize.width, imSize.height);

        // Copy source view to device
        devSrc.copyFrom(HostSrc[index].data, HostSrc[index].pitch);

        // Calculate relative rotation and translation:
        RelativeMatrices(Rrel, trel, HostRef.R, HostRef.t, HostSrc[index].R, HostSrc[index].t);
        tr.row(2) = trel;
        tr = tr.trans();

        // For each depth calculate NCC and update depthmap as required
        for (float d = znear; d <= zfar; d += dstep){
            // Calculate homography:
            H = K * (Rrel + tr / d) * invK;
            H = H / H(2,2);

            // Calculate transformed pixel coordinates
            transform_indexes(devx.data(), devy.data(), H, imSize.width, imSize.height, blocks, threads);

            // interpolate pixel values:
            bilinear_interpolation(devWarped.data(), devSrc.data(),
                                   devx.data(), devy.data(),
                                   devSrc.width(), devSrc.height(),
                                   devx.width(), devx.height(),
                                   blocks, threads);

            // We have no more use for devx and devy, we can use them to store intermediate results now
            // devx - will hold windowed mean of warped image
            // devy - will hold windowed std of warped image
            windowed_mean_column(devInter1.data(), devWarped.data(), winsize, false, imSize.width, imSize.height,
                                 blocks, threads);
            windowed_mean_row(devx.data(), devInter1.data(), winsize, false, imSize.width, imSize.height,
                              blocks, threads);

            windowed_mean_column(devy.data(), devWarped.data(), winsize, true, imSize.width, imSize.height,
                                 blocks, threads);
            windowed_mean_row(devInter1.data(), devy.data(), winsize, false, imSize.width, imSize.height,
                              blocks, threads);

            calculate_STD(devy.data(), devx.data(), devInter1.data(),
                          imSize.width, imSize.height, blocks, threads);

            // calculate NCC for each window which is given by
            // NCC = (mean of products - product of means) / product of standard deviations
            element_multiply(devInter1.data(), Ref, devWarped.data(), imSize.width, imSize.height, blocks, threads);
            windowed_mean_column(devWarped.data(), devInter1.data(), winsize, false, imSize.width, imSize.height,
                                 blocks, threads);
            windowed_mean_row(devInter1.data(), devWarped.data(), winsize, false, imSize.width, imSize.height,
                              blocks, threads);
            calcNCC(devNCC.data(), devInter1.data(),
                    Refmean, devx.data(),
                    Refstd, devy.data(),
                    stdthresh, stdthresh,
                    imSize.width, imSize.height,
                    blocks, threads);

            // only keep depth and bestncc values for which best ncc is greater than current
            // set other values to current ncc and depth
            update_arrays(devDepth.data(), devbestNCC.data(),
                          devNCC.data(), d, imSize.width, imSize.height,
                          blocks, threads);

        }

        // Sum depth results for later averaging, lock the function to avoid multithreading conflicts
        mutex.lock();
        sum_depthmap_NCC(globDepth, globN,
                         devDepth.data(), devbestNCC.data(),
                         nccthresh, imSize.width, imSize.height,
                         blocks, threads);
        mutex.unlock();

        // Free up resources
        nppiFree(devInter1.data());
        nppiFree(devSrc.data());
        nppiFree(devNCC.data());
        nppiFree(devbestNCC.data());
        nppiFree(devDepth.data());
        nppiFree(devx.data());
        nppiFree(devy.data());
        nppiFree(devWarped.data());

        return;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaReset();
        return;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        return;
    }
}

bool PlaneSweep::Denoise(unsigned int niter, double lambda)
{
#ifdef OpenCV_FOUND
    if (depthavailable){
        depthmap8udenoised.setSize(depthmap.width, depthmap.height);
        std::vector<cv::Mat> raw(1);
        raw[0] = cv::Mat(depthmap.height, depthmap.width, CV_8UC1, depthmap8u.data, depthmap8u.pitch);
        cv::Mat out(depthmap.height, depthmap.width, CV_8UC1, depthmap8udenoised.data, depthmap8udenoised.pitch);
        raw[0].data[1] = depthmap8u.data[1];
        cv::denoise_TVL1(raw, out, lambda, niter);
        return true;
    }
#elif
    std::cerr << "\nWarning: OpenCV was not found. Denoising aborted.\n\n";
#endif
    return false;
}

void PlaneSweep::ConvertDepthtoUChar(const camImage<float>& input, camImage<uchar>& output)
{
    output.setSize(input.width, input.height);
    for (size_t x = 0; x < input.width; ++x)
        for (size_t y = 0; y < input.height; ++y)
        {
            int i = x + y * input.width;
            // Check if QNAN
            if (input.data[i] == input.data[i]) output.data[i] = uchar(UCHAR_MAX * std::min(std::max((input.data[i] - znear) / (zfar - znear), 0.f), 1.f));
            else output.data[i] = UCHAR_MAX;
        }
}

PlaneSweep::~PlaneSweep()
{
    cudaReset();
}

bool PlaneSweep::CudaDenoise(int argc, char ** argv, const unsigned int niters, const double lambda, const double tau,
                             const double sigma, const double theta, const double beta, const double gamma)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Starting TVL1 denoising...\n\n");

    if (depthavailable) try
    {

        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaReset();
            return false;
        }

        if (printfNPPinfo() == false)
        {
            cudaReset();
            return false;
        }

        int h = depthmap.height, w = depthmap.width;
        cudaError_t err = cudaFree(d_depthmap);
        if (err != cudaSuccess) std::cerr << cudaGetErrorString(err) << std::endl;
        NPP_CHECK_CUDA(err);
        size_t pitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&d_depthmap, &pitch, w * sizeof(float), h));

        if (threads.x * threads.y == 0) threads = dim3(DEFAULT_BLOCK_XDIM, maxThreadsPerBlock/DEFAULT_BLOCK_XDIM);
        blocks = dim3(ceil(w/(float)threads.x), ceil(h/(float)threads.y));

        depthmapdenoised.setSize(w, h);
        depthmap8udenoised.setSize(w, h);

        npp::ImageNPP_32f_C1 R(w, h);
        npp::ImageNPP_32f_C1 Px(w,h);
        npp::ImageNPP_32f_C1 Py(w,h);
        npp::ImageNPP_32f_C1 rawInput(w,h);
        npp::ImageNPP_32f_C1 T11(w,h), T12(w,h), T21(w,h), T22(w,h), ref(w,h);

        ref.copyFrom(HostRef.data, HostRef.pitch);
        NPP_CHECK_CUDA(cudaMemcpy2D(d_depthmap, pitch, depthmap.data, depthmap.pitch, w * sizeof(float), h, cudaMemcpyHostToDevice));
        rawInput.copyFrom(depthmap.data, depthmap.pitch);

        element_scale(ref.data(), 1/255.f, w, h, blocks, threads);
        Anisotropic_diffusion_tensor(T11.data(), T12.data(), T21.data(), T22.data(), ref.data(), beta, gamma, w, h, blocks, threads);

        element_add(d_depthmap, -znear, w, h, blocks, threads);
        element_add(rawInput.data(), -znear, w, h, blocks, threads);

        double xscale = 1.f/(zfar - znear);
        double inputscale = -sigma/(zfar - znear);

        element_scale(d_depthmap, xscale, w, h, blocks, threads);
        element_scale(rawInput.data(), inputscale, w, h, blocks, threads);

        for (unsigned int i = 0; i < niters; i++){
            double currsigma = i == 0 ? 1 + sigma : sigma;
            denoising_TVL1_calculateP_tensor_weighed(Px.data(), Py.data(), T11.data(), T12.data(), T21.data(), T22.data(),
                                                     d_depthmap, currsigma, w, h, blocks, threads);
            denoising_TVL1_update(d_depthmap, R.data(), Px.data(), Py.data(), rawInput.data(),
                                  tau, theta, lambda, sigma,
                                  w, h, blocks, threads);
        }

        element_scale(d_depthmap, (zfar - znear), w, h, blocks, threads);
        element_add(d_depthmap, znear, w, h, blocks, threads);

        NPP_CHECK_CUDA(cudaMemcpy2D(depthmapdenoised.data, depthmapdenoised.pitch, d_depthmap, pitch, w * sizeof(float), h, cudaMemcpyDeviceToHost));
        ConvertDepthtoUChar(depthmapdenoised, depthmap8udenoised);


        nppiFree(R.data());
        nppiFree(Px.data());
        nppiFree(Py.data());
        nppiFree(rawInput.data());
        nppiFree(T11.data());
        nppiFree(T12.data());
        nppiFree(T22.data());
        nppiFree(T21.data());
        nppiFree(ref.data());

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the TVL1 denoising to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";
        std::cout.flush();

        return true;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaReset();
        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        return false;
    }

    return false;
}

bool PlaneSweep::TGV(int argc, char **argv, const unsigned int niters, const unsigned int warps, const double lambda,
                     const double alpha0, const double alpha1, const double tau, const double sigma, const double beta, const double gamma)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\nStarting TGV...\n\n");

    try
    {

        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaReset();
            return false;
        }

        if (printfNPPinfo() == false)
        {
            cudaReset();
            return false;
        }

        int h = HostRef.height, w = HostRef.width;
        depthmapTGV.setSize(w,h);
        depthmap8uTGV.setSize(w,h);

        if (threads.x * threads.y == 0) threads = dim3(DEFAULT_BLOCK_XDIM, maxThreadsPerBlock/DEFAULT_BLOCK_XDIM);
        blocks = dim3(ceil(w/(float)threads.x), ceil(h/(float)threads.y));

        // Initialize data images:
        npp::ImageNPP_32f_C1 Ref(w,h), Px(w,h), Py(w,h), u(w,h), u0(w,h), u1x(w,h), u1y(w,h), ubar(w,h),
                u1xbar(w,h), u1ybar(w,h), qx(w,h), qy(w,h), qz(w,h), qw(w,h), prodsum(w,h),
                x(w,h), y(w,h), X(w,h), Y(w,h), Z(w,h), dX(w,h), dY(w,h), dZ(w,h), dfx(w,h), dfy(w,h),
                T1(w,h), T2(w,h), T3(w,h), T4(w,h);

        int nimages = std::min(std::max((int)numberimages, 1), (int)HostSrc.size());

        std::vector<npp::ImageNPP_32f_C1> Src(nimages), It(nimages), Iu(nimages), r(nimages);

        // Set initial values for depthmap:
        set_value(u.data(), 1.f, w, h, blocks, threads);
        ubar = u;

        // Copy reference image to device memory and normalize
        Ref.copyFrom(HostRef.data, HostRef.pitch);
        element_scale(Ref.data(), 1/255.f, w, h, blocks, threads);
        Anisotropic_diffusion_tensor(T1.data(), T2.data(), T3.data(), T4.data(), Ref.data(), beta, gamma, w, h, blocks, threads);

        // Matrix storages:
        std::vector<Matrix3D> Rrel(nimages);
        std::vector<Vector3D> Trel(nimages);

        double fx = K(0,0), fy = K(1,1);

        for (int i = 0; i < nimages; i++){
            Src[i] = npp::ImageNPP_32f_C1(w,h);
            It[i] = npp::ImageNPP_32f_C1(w,h);
            r[i] = npp::ImageNPP_32f_C1(w,h);
            Iu[i] = npp::ImageNPP_32f_C1(w,h);

            // Copy source image to device memory and normalize
            Src[i].copyFrom(HostSrc[i].data, HostSrc[i].pitch);
            element_scale(Src[i].data(), 1/255.f, w, h, blocks, threads);

            // Calculate relative rotation and translation
            RelativeMatrices(Rrel[i], Trel[i], HostRef.R, HostRef.t, HostSrc[i].R, HostSrc[i].t);
        }

        for (int l = 0; l < warps; l++){

            // Set last solution as initialization for new level of iterations
            u0 = u;
            ubar = u;

            // Reset variables
            set_value(Px.data(), 0.f, w, h, blocks, threads);
            set_value(Py.data(), 0.f, w, h, blocks, threads);
            set_value(qx.data(), 0.f, w, h, blocks, threads);
            set_value(qy.data(), 0.f, w, h, blocks, threads);
            set_value(qz.data(), 0.f, w, h, blocks, threads);
            set_value(qw.data(), 0.f, w, h, blocks, threads);
            set_value(u1x.data(), 0.f, w, h, blocks, threads);
            set_value(u1y.data(), 0.f, w, h, blocks, threads);
            set_value(u1xbar.data(), 0.f, w, h, blocks, threads);
            set_value(u1ybar.data(), 0.f, w, h, blocks, threads);

            for (int i = 0; i < nimages; i++){

                // Calculate transformed coordinates at u0
                TGV2_transform_coordinates(x.data(), y.data(), X.data(), Y.data(), Z.data(), u0.data(), K, Rrel[i], Trel[i], invK, w, h, blocks, threads);

                // Calculate coordinate derivatives
                TGV2_calculate_coordinate_derivatives(dX.data(), dY.data(), dZ.data(), invK, Rrel[i], w, h, blocks, threads);

                // Calculate f(x,u) derivative wrt u at u0
                TGV2_calculate_derivativeF(dfx.data(), dfy.data(), X.data(), dX.data(), Y.data(), dY.data(), Z.data(), dZ.data(),
                                           fx, fy, w, h, blocks, threads);

                // Interpolate source view at calculated coordinates, giving I(f(x,u0))
                bilinear_interpolation(X.data(), Src[i].data(), x.data(), y.data(), w, h, w, h, blocks, threads);

                // Calculate Iu
                TGV2_calculate_Iu(Iu[i].data(), X.data(), dfx.data(), dfy.data(), w, h, blocks, threads);

                // Subtract reference image from interpolated one giving It
                subtract(It[i].data(), X.data(), Ref.data(), w, h, blocks, threads);

                // Reset r
                set_value(r[i].data(), 0.f, w, h, blocks, threads);
            }

            for (int i = 0; i < niters; i++){

                // Update p values
                TGV2_updateP_tensor_weighed(Px.data(), Py.data(), T1.data(), T2.data(), T3.data(), T4.data(),
                                            ubar.data(), u1xbar.data(), u1ybar.data(), alpha1, sigma, w, h, blocks, threads);

                // Update Q values
                TGV2_updateQ(qx.data(), qy.data(), qz.data(), qw.data(), u1xbar.data(), u1ybar.data(), alpha0,
                             sigma, w, h, blocks, threads);

                // Reset prodsum to 0
                set_value(prodsum.data(), 0.f, w, h, blocks, threads);

                // Iterate over each source view
                for (int j = 0; j < nimages; j++){
                    // Update r values
                    TGV2_updateR(r[j].data(), prodsum.data(), u.data(), u0.data(), It[j].data(), Iu[j].data(), sigma, lambda, w, h, blocks, threads);
                }

                // Update all u values
                TGV2_updateU_tensor_weighed(u.data(), u1x.data(), u1y.data(), T1.data(), T2.data(), T3.data(), T4.data(),
                                            ubar.data(), u1xbar.data(), u1ybar.data(), Px.data(), Py.data(),
                                            qx.data(), qy.data(), qz.data(), qw.data(), prodsum.data(),
                                            alpha0, alpha1, tau, lambda, w, h, blocks, threads);
            }
        }

        // Copy result to host memory
        u.copyTo(depthmapTGV.data, depthmapTGV.pitch);

        // Convert to uchar so it can be easily displayed as gray image
        ConvertDepthtoUChar(depthmapTGV, depthmap8uTGV);

        // Free up resources
        nppiFree(Ref.data());
        nppiFree(Px.data());
        nppiFree(Py.data());
        nppiFree(u.data());
        nppiFree(u0.data());
        nppiFree(u1x.data());
        nppiFree(u1y.data());
        nppiFree(ubar.data());
        nppiFree(u1xbar.data());
        nppiFree(u1ybar.data());
        nppiFree(qx.data());
        nppiFree(qy.data());
        nppiFree(qw.data());
        nppiFree(qz.data());
        nppiFree(prodsum.data());
        nppiFree(x.data());
        nppiFree(y.data());
        nppiFree(X.data());
        nppiFree(Y.data());
        nppiFree(Z.data());
        nppiFree(dX.data());
        nppiFree(dY.data());
        nppiFree(dZ.data());
        nppiFree(dfx.data());
        nppiFree(dfy.data());
        nppiFree(T1.data());
        nppiFree(T2.data());
        nppiFree(T3.data());
        nppiFree(T4.data());

        for (int i = 0; i < nimages; i++){
            nppiFree(Src[i].data());
            nppiFree(It[i].data());
            nppiFree(Iu[i].data());
            nppiFree(r[i].data());
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the TGV to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";
        std::cout.flush();

        return true;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaReset();
        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        return false;

    }

    return false;
}

void PlaneSweep::RelativeMatrices(Matrix3D & Rrel, Vector3D & trel, const Matrix3D & Rref,
                                  const Vector3D & tref, const Matrix3D & Rsrc, const Vector3D & tsrc) const
{
    if (!alternativemethod){
        Rrel = Rsrc * Rref.inv();
        trel = tsrc - Rrel * tref;
    }
    else {
        Rrel = Rsrc.trans() * Rref;
        trel = Rsrc.trans() * (tref - tsrc);
    }
}

void PlaneSweep::get3Dcoordinates(camImage<float> *&x, camImage<float> *&y, camImage<float> *&z)
{
    // calculate reference -> world transformation matrices
    Matrix3D Rr, I;
    Vector3D t, T(0,0,0);
    I.makeIdentity();
    RelativeMatrices(Rr, t, HostRef.R, HostRef.t, I, T);

    int w = HostRef.width, h = HostRef.height;
    npp::ImageNPP_32f_C1 Px(w,h), Py(w,h), X(w,h);

    // copy depthmap to device
    X.copyFrom(depthmapdenoised.data, depthmapdenoised.pitch);

    // calculate world coordinates
    compute3D(Px.data(), Py.data(), X.data(), Rr, t, invK, w, h, blocks, threads);

    // copy coordinates to host images
    coord_x.setSize(w, h);
    coord_y.setSize(w, h);
    coord_z.setSize(w, h);
    Px.copyTo(coord_x.data, coord_x.pitch);
    Py.copyTo(coord_y.data, coord_y.pitch);
    X.copyTo(coord_z.data, coord_z.pitch);
    x = &coord_x; y = &coord_y; z = &coord_z;

    // free resources
    nppiFree(X.data());
    nppiFree(Px.data());
    nppiFree(Py.data());
}

void PlaneSweep::cudaReset()
{
    NPP_CHECK_CUDA(cudaDeviceReset());

    // set pointers to NULL so cudaFree will not try to free wrong memory
    d_depthmap = 0;
}

bool PlaneSweep::TGVdenoiseFromSparse(int argc, char **argv, const camImage<float> & depth, const unsigned int niters,
                                      const double alpha0, const double alpha1, const double tau, const double sigma, const double theta,
                                      const double beta, const double gamma)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("\nStarting TGV denoising...\n\n");

    try
    {

        if (cudaDevInit(argc, (const char **)argv) == NO_CUDA_DEVICE)
        {
            cudaReset();
            return false;
        }

        if (printfNPPinfo() == false)
        {
            cudaReset();
            return false;
        }

        int h = HostRef.height, w = HostRef.width;
        depthmapTGV.setSize(w, h);

        npp::ImageNPP_32f_C1 px(w,h), py(w,h), qx(w,h), qy(w,h), qz(w,h), qw(w,h), /*u(w,h),*/ ubar(w,h),
                vx(w,h), vy(w,h), vxbar(w,h), vybar(w,h), weights(w,h), Ds(w,h), ref(w,h), T1(w,h), T2(w,h), T3(w,h), T4(w,h);

        cudaError_t err = cudaFree(d_depthmap);
        if (err != cudaSuccess) std::cerr << cudaGetErrorString(err) << std::endl;
        NPP_CHECK_CUDA(err);
        size_t pitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&d_depthmap, &pitch, w * sizeof(float), h));

        if (threads.x * threads.y == 0) threads = dim3(DEFAULT_BLOCK_XDIM, maxThreadsPerBlock/DEFAULT_BLOCK_XDIM);
        blocks = dim3(ceil(w/(float)threads.x), ceil(h/(float)threads.y));

        Ds.copyFrom(depth.data, depth.pitch);
        calculateWeights_sparseDepth(weights.data(), Ds.data(), w, h, blocks, threads);
        element_scale(Ds.data(), 1.f / zfar, w, h, blocks, threads);
        ubar.copyFrom(depthmap.data, depthmap.pitch);
        element_scale(ubar.data(), 1.f / zfar, w, h, blocks, threads);
//        ubar = u;
        NPP_CHECK_CUDA(cudaMemcpy2D(d_depthmap, pitch, ubar.data(), ubar.pitch(), w * sizeof(float), h, cudaMemcpyDeviceToDevice));

        ref.copyFrom(HostRef.data, HostRef.pitch);
        element_scale(ref.data(), 1.f / 255.f, w, h, blocks, threads);

        Anisotropic_diffusion_tensor(T1.data(), T2.data(), T3.data(), T4.data(), ref.data(), beta, gamma, w, h, blocks, threads);
//        set_value(T1.data(), 1.f, w, h, blocks, threads);
//        set_value(T3.data(), 1.f, w, h, blocks, threads);

        for (int i = 0; i < niters; i++){
            TGV2_updateP_tensor_weighed(px.data(), py.data(), T1.data(), T2.data(), T3.data(), T4.data(),
                                        d_depthmap, vxbar.data(), vybar.data(), alpha1, sigma, w, h, blocks, threads);
            TGV2_updateQ(qx.data(), qy.data(), qz.data(), qw.data(), vxbar.data(), vybar.data(), alpha0, sigma, w, h, blocks, threads);
            TGV2_updateU_sparseDepthTensor(d_depthmap, vxbar.data(), vybar.data(), ubar.data(), vxbar.data(), vybar.data(),
                                           T1.data(), T2.data(), T3.data(), T4.data(), px.data(), py.data(),
                                           qx.data(), qy.data(), qz.data(), qw.data(), weights.data(), Ds.data(), alpha0, alpha1, tau, theta, w, h,
                                           blocks, threads);
        }

        element_scale(d_depthmap, zfar, w, h, blocks, threads);
        //ubar.copyTo(depthmapTGV.data, depthmapTGV.pitch);
        NPP_CHECK_CUDA(cudaMemcpy2D(depthmapTGV.data, depthmapTGV.pitch, d_depthmap, pitch, w * sizeof(float), h, cudaMemcpyDeviceToHost));
        ConvertDepthtoUChar(depthmapTGV, depthmap8uTGV);

        nppiFree(px.data());
        nppiFree(py.data());
        nppiFree(qx.data());
        nppiFree(qy.data());
        nppiFree(qz.data());
        nppiFree(qw.data());
        //(u.data());
        nppiFree(ubar.data());
        nppiFree(vx.data());
        nppiFree(vy.data());
        nppiFree(vxbar.data());
        nppiFree(vybar.data());
        nppiFree(weights.data());
        nppiFree(Ds.data());
        nppiFree(ref.data());
        nppiFree(T1.data());
        nppiFree(T2.data());
        nppiFree(T3.data());
        nppiFree(T4.data());

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken for the TGV to complete is " <<
                     std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms\n\n";
        std::cout.flush();

        return true;

    }
    catch (npp::Exception &rExcep)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rExcep << std::endl;
        std::cerr << "Aborting." << std::endl;

        cudaReset();
        return false;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        return false;
    }

    return false;
}
