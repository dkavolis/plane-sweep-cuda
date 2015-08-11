/**
 *  \file defines.h
 *  \brief Header file containing fusion and planesweep defines
 */
#ifndef DEFINES_H
#define DEFINES_H

// synthetic data properties file variable names
#define CAM_POS                     "cam_pos"
#define CAM_DIR                     "cam_dir"
#define CAM_UP                      "cam_up"
#define CAM_LOOKAT                  "cam_lookat"
#define CAM_SKY                     "cam_sky"
#define CAM_RIGHT                   "cam_right"
#define CAM_FPOINT                  "cam_fpoint"
#define CAM_ANGLE                   "cam_angle"

// RGB to grayscale conversion weights
#define RGB2GRAY_WEIGHT_RED         0.2989
#define RGB2GRAY_WEIGHT_GREEN       0.5870
#define RGB2GRAY_WEIGHT_BLUE        0.1140

// Default planesweep parameters
#define DEFAULT_Z_NEAR              0.1f
#define DEFAULT_Z_FAR               1.0f
#define DEFAULT_NUMBER_OF_PLANES    200
#define DEFAULT_NUMBER_OF_IMAGES    4
#define DEFAULT_WINDOW_SIZE         5
#define DEFAULT_STD_THRESHOLD       0.0001f
#define DEFAULT_NCC_THRESHOLD       0.5f
#define NO_DEPTH                    -1

// Default GPU parameters
#define NO_CUDA_DEVICE              -1
#define MAX_THREADS_PER_BLOCK       512
#define MAX_PLANESWEEP_THREADS      1 // multithreading does not reduce execution time
#define DEFAULT_BLOCK_XDIM          32

// Default TVL1 denoising parameters
#define DEFAULT_TVL1_ITERATIONS     100
#define DEFAULT_TVL1_LAMBDA         .3
#define DEFAULT_TVL1_TAU            0.02
#define DEFAULT_TVL1_SIGMA          6.f
#define DEFAULT_TVL1_THETA          1.f
#define DEFAULT_TVL1_BETA           0.f
#define DEFAULT_TVL1_GAMMA          1.f

// Default TGV2 parameters
#define DEFAULT_TGV_LAMBDA          0.5
#define DEFAULT_TGV_ALPHA0          2.0
#define DEFAULT_TGV_ALPHA1          1.5
#define DEFAULT_TGV_NITERS          30
#define DEFAULT_TGV_NWARPS          15
#define DEFAULT_TGV_SIGMA           1.f
#define DEFAULT_TGV_TAU             0.02
#define DEFAULT_TGV_BETA            0.f
#define DEFAULT_TGV_GAMMA           1.f

// Default fusion parameters
#define DEFAULT_FUSION_SD_THRESHOLD 0.05
#define DEFAULT_FUSION_TAU          0.1
#define DEFAULT_FUSION_LAMBDA       0.5
#define DEFAULT_FUSION_SIGMA        1.f
#define DEFAULT_FUSION_IMSTEP       3
#define DEFAULT_FUSION_ITERATIONS   50

// Default fusion kernel parameters
#define DEFAULT_FUSION_THREADS_X    16
#define DEFAULT_FUSION_THREADS_Y    16

// Default fusion voxel numbers
#define DEFAULT_FUSION_VOXELS_X     448
#define DEFAULT_FUSION_VOXELS_Y     336
#define DEFAULT_FUSION_VOXELS_Z     160

// Default fusion volume (living room set)
#define DEFAULT_FUSION_VOLUME_X1    -2.71
#define DEFAULT_FUSION_VOLUME_Y1    -.10
#define DEFAULT_FUSION_VOLUME_Z1    -4.8
#define DEFAULT_FUSION_VOLUME_X2    2.79
#define DEFAULT_FUSION_VOLUME_Y2    2.9
#define DEFAULT_FUSION_VOLUME_Z2    4.2

#endif // DEFINES_H
