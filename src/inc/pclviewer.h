/**
 *  \file pclviewer.h
 *  \brief Header file containing GUI controlling class
 */
#ifndef PCLVIEWER_H
#define PCLVIEWER_H

#define CAM_POS "cam_pos"
#define CAM_DIR "cam_dir"
#define CAM_UP "cam_up"
#define CAM_LOOKAT "cam_lookat"
#define CAM_SKY "cam_sky"
#define CAM_RIGHT "cam_right"
#define CAM_FPOINT "cam_fpoint"
#define CAM_ANGLE "cam_angle"

#define RGB2GRAY_WEIGHT_RED 0.2989
#define RGB2GRAY_WEIGHT_GREEN 0.5870
#define RGB2GRAY_WEIGHT_BLUE 0.1140

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

#include <planesweep.h>
#include "fusion.cu.h"

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
	 *  
	 *  \details
	 */
  PCLViewer (int argc, char **argv, QWidget *parent = 0);
  
	/**
	 *  \brief Default destructor
	 */
  ~PCLViewer ();

  /**
   *  \brief Set command line arguments
   *  
   *  \param ac number of arguments
   *  \param av pointer to command line argument strings
   *  \return No return value
   *  
   *  \details
   */
  void setArgs(int ac, char **av){ argc = ac; argv = av; }

protected:
// PCLVisualizer pointers, one for each qvtkwidget
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerdenoised;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> tgvviewer;
  
  // Point cloud pointers, one for each qvtkwidget
  PointCloudT::Ptr cloud;
  PointCloudT::Ptr clouddenoised;
  PointCloudT::Ptr tgvcloud;

  // command line arguments
  int argc;
  char **argv;

  // classes that implement 3d reconstruction methods
  PlaneSweep ps;
  dfusionData8 fd;

private slots: // GUI widget slots
  void pSliderValueChanged (int value);
	
  void on_pushButton_pressed();

  void on_imNumber_valueChanged(int arg1);

  void on_winSize_valueChanged(int arg1);

  void on_numberPlanes_valueChanged(int arg1);

  void on_zNear_valueChanged(double arg1);

  void on_zFar_valueChanged(double arg1);

  void on_stdThresh_valueChanged(double arg1);

  void on_nccThresh_valueChanged(double arg1);

  void on_pSlider2_valueChanged(int value);

  void on_denoiseBtn_clicked();

  void on_threadsx_valueChanged(int arg1);

  void on_threadsy_valueChanged(int arg1);

  void on_tgv_button_pressed();

  void on_tgv_psize_valueChanged(int value);

  void on_imagePathButton_clicked();

  void on_imageNameButton_clicked();

  void on_altmethod_toggled(bool checked);

  void on_loadfromsrc_clicked();

  void on_loadfromdir_clicked();

  void on_save_clicked();

  void on_nIters_valueChanged(int arg1);

  void on_lambda_valueChanged(double arg1);

  void on_tvl1_tau_valueChanged(double arg1);

  void on_tvl1_sigma_valueChanged(double arg1);

  void on_tvl1_theta_valueChanged(double arg1);

  void on_tvl1_beta_valueChanged(double arg1);

  void on_tvl1_gamma_valueChanged(double arg1);

  void on_reconstruct_button_clicked();

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

  // variables to track if reference image has changed before last depthmap generation
  // for each method
  bool  refchanged = true,
        refchangedtvl1 = true,
        refchangedtgv = true;

  	/**
   *  \brief Load images from source directory
   *  
   *  \return No return value
   *  
   *  \details Images are assumed to be at \a "../src/PlaneSweep/". This requires build directory to be in the same folder as 
   *  source directory. 
   */
  void LoadImages();
  
  /**
   *  \brief Concatenate strings from UI and given \a number to create image file name
   *  
   *  \param number   image number
   *  \param imagePos camera parameter file for this image returned by reference
   *  \return Image file name
   *  
   *  \details
   */
  QString ImageName(int number, QString & imagePos);
  
  /**
   *  \brief Compute camera calibration matrix \f$K\f$
   *  
   *  \param K         camera calibration matrix \f$K\f$
   *  \param cam_dir   camera parameter
   *  \param cam_up    camera parameter
   *  \param cam_right camera parameter
   *  \return No return value
   *  
   *  \details Camera parameters must first be obtained via getcamParameters
   */
  void getcamK(ublas::matrix<double> & K, const ublas::matrix<double> & cam_dir,
               const ublas::matrix<double> & cam_up, const ublas::matrix<double> & cam_right);
			   
  /**
   *  \brief Compute rotation matrix and translation vector
   *  
   *  \param R       rotation matrix
   *  \param t       translation vector
   *  \param cam_dir camera parameter
   *  \param cam_pos camera parameter
   *  \param cam_up  camera parameter
   *  \return No return value
   *  
   *  \details Camera parameters must first be obtained via getcamParameters
   */
  void computeRT(ublas::matrix<double> & R, ublas::matrix<double> & t, const ublas::matrix<double> & cam_dir,
                 const ublas::matrix<double> & cam_pos, const ublas::matrix<double> & cam_up);
  
  /**
   *  \brief Get cam parameters from \a txt file
   *  
   *  \param filename   name of \a .txt file, including extension
   *  \param cam_pos    camera parameter
   *  \param cam_dir    camera parameter
   *  \param cam_up     camera parameter
   *  \param cam_lookat camera parameter
   *  \param cam_sky    camera parameter
   *  \param cam_right  camera parameter
   *  \param cam_fpoint camera parameter
   *  \param cam_angle  camera parameter
   *  \return Success/failure of opening \a filename
   *  
   *  \details Must be the same format as ones found on http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
   */
  bool getcamParameters(QString filename, ublas::matrix<double> & cam_pos, ublas::matrix<double> & cam_dir,
                        ublas::matrix<double> & cam_up, ublas::matrix<double> & cam_lookat,
                        ublas::matrix<double> & cam_sky, ublas::matrix<double> & cam_right,
                        ublas::matrix<double> & cam_fpoint, double & cam_angle);
						
  /**
   *  \brief Compute vector product of 2 \a boost matrices
   *  
   *  \param A \f$1^{st}\f$ vector
   *  \param B \f$2^{nd}\f$ vector
   *  \return Vector product of \a A and \a B
   *  
   *  \details Both matrices must be of size (1,2), (1,3), (2,1) or (3,1).
   */
  ublas::matrix<double> &cross(const ublas::matrix<double> & A, const ublas::matrix<double> & B);

  /**
   *  \brief RGB Qimage to grayscale conversion using predefined colour weights
   *  
   *  \tparam T type of data to convert to 
   *  \param data pointer to output data
   *  \param img  RGB image to convert
   *  \return No return value
   *  
   *  \details Weights are defined in preprocessor definitions
   */
  template<typename T>
  void rgb2gray(T * data, const QImage & img);
};

 /** @} */ // group gui

#endif // PCLVIEWER_H
