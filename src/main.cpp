#include <QApplication>
#include <QMainWindow>
#include <pclviewer.h>
//#include <D:/Software/Visual Leak Detector/include/vld.h>

int main (int argc, char *argv[])
{
  QApplication a (argc, argv);
  PCLViewer w(argc, (char **)argv);
  w.show();
  w.setArgs(argc, (char **)argv);

  return a.exec ();
}
