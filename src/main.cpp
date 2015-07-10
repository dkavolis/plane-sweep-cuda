#include <QApplication>
#include <QMainWindow>
#include <pclviewer.h>

int main (int argc, char *argv[])
{
  QApplication a (argc, argv);
  PCLViewer w(argc, (char **)argv);
  w.show();
  w.setArgs(argc, (char **)argv);

  return a.exec ();
}
