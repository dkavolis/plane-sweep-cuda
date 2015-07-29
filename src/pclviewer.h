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

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

namespace Ui
{
  class PCLViewer;
}

class PCLViewer : public QMainWindow
{
  Q_OBJECT

public:
  PCLViewer (int argc, char **argv, QWidget *parent = 0);
  ~PCLViewer ();

  void setArgs(int ac, char **av){ argc = ac; argv = av; }

public slots:

  void
  pSliderValueChanged (int value);

protected:
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewerdenoised;
  boost::shared_ptr<pcl::visualization::PCLVisualizer> tgvviewer;
  PointCloudT::Ptr cloud;
  PointCloudT::Ptr clouddenoised;
  PointCloudT::Ptr tgvcloud;

  unsigned int red;
  unsigned int green;
  unsigned int blue;

  unsigned int nimages;
  unsigned int refnumber;
  unsigned int winsize;
  float znear;
  float zfar;
  unsigned int planenumber;
  double nccthresh;
  double stdthresh;
  unsigned char ndigits;

  int argc;
  char **argv;

  QString impath;
  QString imformat;
  QString imname;

  PlaneSweep ps;

private slots:
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

private:
  Ui::PCLViewer *ui;
  void LoadImages();
  QPixmap image;
  QImage  refim;
  QImage refgray;
  std::vector<QImage> sources;
  QGraphicsScene *scene;
  QGraphicsScene *depthscene;
  QGraphicsScene *dendepthsc;
  QGraphicsScene *tgvscene;
  QPixmap depthim;
  QPixmap dendepthim;
  QPixmap tgvdepthim;
  PlaneSweep::camImage<float> * depth;
  PlaneSweep::camImage<uchar> * depth8u;

  PlaneSweep::camImage<float> * dendepth;
  PlaneSweep::camImage<uchar> * dendepth8u;

  PlaneSweep::camImage<float> * tgvdepth;
  PlaneSweep::camImage<uchar> * tgvdepth8u;

  bool  refchanged = true,
        refchangedtvl1 = true,
        refchangedtgv = true;

  QString ImageName(int number, QString & imagePos);
  void getcamK(ublas::matrix<double> & K, const ublas::matrix<double> & cam_dir,
               const ublas::matrix<double> & cam_up, const ublas::matrix<double> & cam_right);
  void computeRT(ublas::matrix<double> & R, ublas::matrix<double> & t, const ublas::matrix<double> & cam_dir,
                 const ublas::matrix<double> & cam_pos, const ublas::matrix<double> & cam_up);
  bool getcamParameters(QString filename, ublas::matrix<double> & cam_pos, ublas::matrix<double> & cam_dir,
                        ublas::matrix<double> & cam_up, ublas::matrix<double> & cam_lookat,
                        ublas::matrix<double> & cam_sky, ublas::matrix<double> & cam_right,
                        ublas::matrix<double> & cam_fpoint, double & cam_angle);
  ublas::matrix<double> &cross(const ublas::matrix<double> & A, const ublas::matrix<double> & B);

  template<typename T>
  void rgb2gray(T * data, const QImage & img);


};

#endif // PCLVIEWER_H
