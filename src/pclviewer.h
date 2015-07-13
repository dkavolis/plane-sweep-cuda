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
  PointCloudT::Ptr cloud;

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

private:
  Ui::PCLViewer *ui;
  void LoadImages();
  QPixmap image;
  QImage  refim;
  QImage refgray;
  std::vector<QImage> sources;
  QGraphicsScene *scene;
  PlaneSweep::camImage<float> depth;

};

#endif // PCLVIEWER_H
