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
 *  \brief Update histogram from given \a depthmap
 *
 *  \param f            pointer to \a fusionData containing histogram
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
void FusionUpdateHistogram(fusionData<_bins> * f, const float * depthmap, const Matrix3D & K, const Matrix3D & R, const float3 & t, const float threshold, const int width, const int height, dim3 blocks, dim3 threads);

/**
 *  \brief Update primal variable \f$u\f$ and helper variable \f$v\f$ using histogram depthmap fusion algorithm
 *
 *  \param f       pointer to \a fusionData containing \f$u\f$
 *  \param tau     fusion parameter \f$\tau\f$
 *  \param lambda  fusion parameter \f$\lambda\f$
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *
 *  \details
 */
template<unsigned char _bins>
void FusionUpdateU(fusionData<_bins> * f, const double tau, const double lambda, dim3 blocks, dim3 threads);

/**
 *  \brief Update dual variable \f$p\f$ using histogram depthmap fusion algorithm
 *
 *  \param f       pointer to \a fusionData containing \f$p\f$
 *  \param sigma   fusion parameter \f$\sigma\f$
 *  \param blocks  kernel grid dimensions
 *  \param threads single block dimensions
 *  \return No return value
 *
 *  \details
 */
template<unsigned char _bins>
void FusionUpdateP(fusionData<_bins> * f, const double sigma, dim3 blocks, dim3 threads);

/**
 *  \brief Single depthmap fusion iteration function
 *
 *  \param f            pointer to \a fusionData
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
void FusionUpdateIteration(fusionData<_bins> * f, const float * depthmap, const Matrix3D & K, const Matrix3D & R, const float3 & t, const float threshold, const double tau, const double lambda, const double sigma, const int width, const int height, dim3 blocks, dim3 threads);

/** @} */ // group fusion

#endif // FUSION_CU_H
