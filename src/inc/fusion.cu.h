/**
 *  \file fusion.cu.h
 *  \brief Header file containing depthmap fusion functions
 */
#ifndef FUSION_CU_H
#define FUSION_CU_H

#include <helper_cuda.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "fusion.h"

/** \addtogroup fusion  Depthmap fusion
* \brief Depthmap fusion functions running on GPU
*
* \details For more info see page 38 of http://gpu4vision.icg.tugraz.at/papers/2012/graber_master.pdf#pub68
*
*  \tparam _bins  number of histogram bins in \a fusionData
*
* @{
*/

/**
 *  \brief Update histogram from given \p depthmap
 *
 *  \param f            \p fusionData containing histogram
 *  \param depthmap     pointer to depthmap data
 *  \param K            3x3 camera calibration matrix \f$K\f$
 *  \param R            3x3 rotation matrix from world to camera coordinates
 *  \param t            translation vector from world to camera position
 *  \param threshold    signed distance value threshold
 *  \param width        width of depthmap
 *  \param height       height of depthmap
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *
 *  \details Signed distance is clamped to [-threshold,threshold] and divided by \a threshold before updating any histogram bins.
 */
template<unsigned char _bins>
void FusionUpdateHistogram(fusionData<_bins> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                           const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Update primal variable \f$u\f$ and helper variable \f$v\f$ using histogram depthmap fusion algorithm
 *
 *  \param f       \p fusionData containing \f$u\f$
 *  \param tau     fusion parameter \f$\tau\f$
 *  \param lambda  fusion parameter \f$\lambda\f$
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *
 *  \details
 */
template<unsigned char _bins>
void FusionUpdateU(fusionData<_bins> f, const double tau, const double lambda, dim3 blocks, dim3 threads);

/**
 *  \brief Update dual variable \f$p\f$ using histogram depthmap fusion algorithm
 *
 *  \param f       \p fusionData containing \f$p\f$
 *  \param sigma   fusion parameter \f$\sigma\f$
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *
 *  \details
 */
template<unsigned char _bins>
void FusionUpdateP(fusionData<_bins> f, const double sigma, dim3 blocks, dim3 threads);

/**
 *  \brief Single depthmap fusion iteration function
 *
 *  \param f            \p fusionData
 *  \param depthmap     pointer to depthmap to be used in updating histograms
 *  \param K            3x3 camera calibration matrix \f$K\f$
 *  \param R            3x3 rotation matrix from world to camera coordinates
 *  \param t            translation vector from world to camera position
 *  \param threshold    signed distance value threshold
 *  \param tau          fusion parameter \f$\tau\f$
 *  \param lambda       fusion parameter \f$\lambda\f$
 *  \param sigma        fusion parameter \f$\sigma\f$
 *  \param width        width of depthmap
 *  \param height       height of depthmap
 *  \param blocks       kernel grid dimensions
 *  \param threads      single block dimensions
 *  \return No return value
 *
 *  \details Signed distance is clamped to [-threshold,threshold] and divided by \a threshold before updating any histogram bins.
 */
template<unsigned char _bins>
void FusionUpdateIteration(fusionData<_bins, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                           const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                           const int width, const int height, dim3 blocks, dim3 threads);

// Explicit template instantiations
template void
FusionUpdateIteration<2>(fusionData<2, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<3>(fusionData<3, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<4>(fusionData<4, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<5>(fusionData<5, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<6>(fusionData<6, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<7>(fusionData<7, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<8>(fusionData<8, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<9>(fusionData<9, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                         const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                         const int width, const int height, dim3 blocks, dim3 threads);
template void
FusionUpdateIteration<10>(fusionData<10, Device> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                          const Vector3D t, const float threshold, const double tau, const double lambda, const double sigma,
                          const int width, const int height, dim3 blocks, dim3 threads);

template void FusionUpdateP<2>(fusionData<2> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<3>(fusionData<3> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<4>(fusionData<4> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<5>(fusionData<5> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<6>(fusionData<6> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<7>(fusionData<7> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<8>(fusionData<8> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<9>(fusionData<9> f, const double sigma, dim3 blocks, dim3 threads);
template void FusionUpdateP<10>(fusionData<10> f, const double sigma, dim3 blocks, dim3 threads);

template void FusionUpdateU<2>(fusionData<2> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<3>(fusionData<3> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<4>(fusionData<4> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<5>(fusionData<5> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<6>(fusionData<6> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<7>(fusionData<7> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<8>(fusionData<8> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<9>(fusionData<9> f, const double tau, const double lambda, dim3 blocks, dim3 threads);
template void FusionUpdateU<10>(fusionData<10> f, const double tau, const double lambda, dim3 blocks, dim3 threads);

template void FusionUpdateHistogram<2>(fusionData<2> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                      const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<3>(fusionData<3> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<4>(fusionData<4> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<5>(fusionData<5> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<6>(fusionData<6> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<7>(fusionData<7> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<8>(fusionData<8> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<9>(fusionData<9> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                       const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
template void FusionUpdateHistogram<10>(fusionData<10> f, const float * depthmap, const Matrix3D K, const Matrix3D R,
                                        const Vector3D t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);
/** @} */ // group fusion

#endif // FUSION_CU_H
