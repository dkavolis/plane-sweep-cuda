/**
 *  \file pclviewer.h
 *  \brief Header file containing GUI controlling class
 */
#ifndef PCLVIEWER_H
#define PCLVIEWER_H

#include "defines.h"
#include <iostream>

// Qt
#include <QMainWindow>
#include <QString>
#include <QImage>
#include <QGraphicsScene>

// Point Cloud Library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Visualization Toolkit (VTK)
#include <vtkRenderWindow.h>
#include <QVTKWidget.h>

#include "planesweep.h"
#include "fusion.cu.h"
#include "kitti_data.h"
#include "reader.h"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui
{
class PCLViewer;
}

/** \addtogroup gui GUI
*  \brief GUI group
* @{
*/

/**
*  \brief Class that controls interactions between GUI and other classes
*/
class PCLViewer : public QMainWindow
{
    Q_OBJECT

public:

    /**
    *  \brief Constructor
    *
    *  \param argc number of command line arguments
    *  \param argv pointer to command line argument strings
    *  \param parent pointer to parent widget
    */
    PCLViewer (int argc, char **argv, QWidget *parent = 0);

    /** \brief Default destructor */
    ~PCLViewer ();

    /**
    *  \brief Set command line arguments
    *
    *  \param ac number of arguments
    *  \param av pointer to command line argument strings
    *  \return No return value
    */
    void setArgs(int ac, char **av){ argc = ac; argv = av; }

protected:
    // PCLVisualizer pointers, one for each qvtkwidget
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerdenoised;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewertgv;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerfusion;

    // Point cloud pointers, one for each qvtkwidget
    PointCloudT::Ptr cloud;
    PointCloudT::Ptr clouddenoised;
    PointCloudT::Ptr cloudtgv;
    PointCloudT::Ptr cloudfusion;

    // command line arguments
    int argc;
    char **argv;

    KITTI::KITTIData kitti;
    Reader reader;

    // classes that implement 3d reconstruction methods
    PlaneSweep ps;
    dfusionData8 fd;
    fusionData<8, Standard> f;

private slots: // GUI widgets slots

    // Fusion volume widgets slots:///////////////////
    void fusion_volume_center_x_changed();

    void fusion_volume_center_y_changed();

    void fusion_volume_center_z_changed();

    void fusion_volume_corner_x_changed();

    void fusion_volume_corner_y_changed();

    void fusion_volume_corner_z_changed();
    ////////////////////////////////////////////

    /** \brief Planesweep QVTK widget point size slider slot */
    void pSliderValueChanged (int value);

    /** \brief Planesweep start push button slot */
    void on_pushButton_pressed();

    // Planesweep settings widgets slots:////////////
    void on_imNumber_valueChanged(int arg1);

    void on_winSize_valueChanged(int arg1);

    void on_numberPlanes_valueChanged(int arg1);

    void on_zNear_valueChanged(double arg1);

    void on_zFar_valueChanged(double arg1);

    void on_stdThresh_valueChanged(double arg1);

    void on_nccThresh_valueChanged(double arg1);

    void on_threadsx_valueChanged(int arg1);

    void on_threadsy_valueChanged(int arg1);

    void on_altmethod_toggled(bool checked);
    ////////////////////////////////////////////

    /** \brief Planesweep + TVL1 denoising QVTK widget point size slider slot */
    void on_pSlider2_valueChanged(int value);

    /** \brief TVL1 denoising on planesweep depthmap push button slot */
    void on_denoiseBtn_clicked();

    /** \brief TGV push button slot */
    void on_tgv_button_pressed();

    /** \brief TGV QVTK widget point size slider slot */
    void on_tgv_psize_valueChanged(int value);

    /** \brief Select path to images push button slot */
    void on_imagePathButton_clicked();

    /** \brief Select reference image push button slot */
    void on_imageNameButton_clicked();

    /** \brief Load images from source push button slot */
    void on_loadfromsrc_clicked();

    /** \brief Load images from selected directory push button slot */
    void on_loadfromdir_clicked();

    /** \brief Save all results push button slot */
    void on_save_clicked();

    // TVL1 denoising widgets slots, only useful if 'denoise on parameter change' is checked:
    void on_nIters_valueChanged(int arg1);

    void on_lambda_valueChanged(double arg1);

    void on_tvl1_tau_valueChanged(double arg1);

    void on_tvl1_sigma_valueChanged(double arg1);

    void on_tvl1_theta_valueChanged(double arg1);

    void on_tvl1_beta_valueChanged(double arg1);

    void on_tvl1_gamma_valueChanged(double arg1);
    //////////////////////////////////////////////////

    /** \brief Planesweep + TVL1 + fusion reconstruction push button slot */
    void on_reconstruct_button_clicked();

    // Fusion widgets slots:////////////////////////////////////////////
    void on_fusion_psize_valueChanged(int value);

    void on_fusion_resize_clicked();

    void on_fusion_threadsw_valueChanged(int arg1);

    void on_fusion_threadsh_valueChanged(int arg1);

    void on_fusion_threadsd_valueChanged(int arg1);
    //////////////////////////////////////////////////////////////////////

    void colorbar_selected(double value);

private:
    // pointer to UI
    Ui::PCLViewer *ui;

    // reference RGB image
    QImage  refim;

    // pointers to scenes, one for each QGraphicsWidget
    QGraphicsScene *scene;
    QGraphicsScene *depthscene;
    QGraphicsScene *dendepthsc;
    QGraphicsScene *tgvscene;

    // pixmaps of images
    QPixmap image;
    QPixmap depthim;
    QPixmap dendepthim;
    QPixmap tgvdepthim;

    // pointers to depthmaps
    PlaneSweep::camImage<float> * depth;
    PlaneSweep::camImage<uchar> * depth8u;

    PlaneSweep::camImage<float> * dendepth;
    PlaneSweep::camImage<uchar> * dendepth8u;

    PlaneSweep::camImage<float> * tgvdepth;
    PlaneSweep::camImage<uchar> * tgvdepth8u;

    // pointers to world coordinates
    PlaneSweep::camImage<float> * cx;
    PlaneSweep::camImage<float> * cy;
    PlaneSweep::camImage<float> * cz;

    // sparse depthmap
    PlaneSweep::camImage<float> sparsedepth;

    // variables to track if reference image has changed before last depthmap generation
    // for each method
    bool  refchanged = true,
    refchangedtvl1 = true,
    refchangedtgv = true;

    QVector<QRgb> ctable;
    /**
    *  \brief Load images from source directory
    *
    *  \details Images are assumed to be source directory.
    */
    void LoadImages();

    /**
    *  \brief Concatenate strings from UI and given \a number to create image file name
    *
    *  \param number   image number
    *  \param imagePos camera parameter file for this image returned by reference
    *  \return Image file name
    */
    QString ImageName(int number, QString & imagePos);


    /**
    *  \brief RGB Qimage to grayscale conversion using predefined colour weights
    *
    *  \tparam T type of data to convert to
    *  \param data pointer to output data
    *  \param img  RGB image to convert
    *
    *  \details Weights are defined in preprocessor definitions
    */
    template<typename T>
    void rgb2gray(T * data, const QImage & img);

    /**
    *  \brief Depthmap coloring function
    *
    *  \param depth normalized depth in range [0, 255]
    *  \return Uchar3 struct containing RGB colors in this order
    */
    uchar3 RGBdepthmapColor(uchar depth);

    /** \brief Function to setup planesweep GUI widgets */
    void setupPlanesweep();

    /** \brief Function to setup TGV GUI widgets */
    void setupTGV();

    /** \brief Function to setup fusion GUI widgets */
    void setupFusion();

    bool loadSparseDepthmap(const QString & fileName);
};

/** @} */ // group gui

#endif // PCLVIEWER_H
