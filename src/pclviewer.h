#ifndef PCLVIEWER_H
#define PCLVIEWER_H

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
  QPixmap dendepth;
  QPixmap tgvdepth;
  PlaneSweep::camImage<float> depth;
  PlaneSweep::camImage<uchar> depth8u;
  PlaneSweep::camImage<uchar> dendepth8u;
  PlaneSweep::camImage<uchar> tgvdepth8u;

};

#endif // PCLVIEWER_H
