#include "pclviewer.h"
#include "../qt build/ui_pclviewer.h"
#include <iostream>
#include <QDir>
#include <QByteArray>
#include <QBuffer>
#include <boost/numeric/ublas/assignment.hpp>
#include <QPixmap>

PCLViewer::PCLViewer (int argc, char **argv, QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::PCLViewer),
    scene (new QGraphicsScene())
{
    ui->setupUi (this);
    this->setWindowTitle ("PCL viewer");

    // Setup the cloud pointer
    cloud.reset (new PointCloudT);
    // The number of points in the cloud
    cloud->points.resize (200);

    // The default color
    red   = 128;
    green = 128;
    blue  = 128;

    // Fill the cloud with some points
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);

        cloud->points[i].r = red;
        cloud->points[i].g = green;
        cloud->points[i].b = blue;
    }

    // Set up the QVTK window
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
    viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
    ui->qvtkWidget->update ();

    connect(ui->pSlider, SIGNAL(valueChanged(int)), this, SLOT(pSliderValueChanged(int)));

    viewer->addPointCloud (cloud, "cloud");
    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();

    LoadImages();
}

void
PCLViewer::pSliderValueChanged (int value)
{
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtkWidget->update ();
}

void PCLViewer::LoadImages()
{
    QString loc = "/../src/PlaneSweep/im";
    QString ref = QDir::currentPath();
    ref += loc;
    ref += QString::number(0);
    ref += ".png";
    refim.load(ref);

    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    double K[3][3] = {
        {0.709874*640, (1-0.977786)*640, 0.493648},
        {0, 0.945744*480, 0.514782*480},
        {0, 0, 1}
    };

    ps.setK(K);
    ublas::matrix<double> Cr(3,4);
    std::vector<ublas::matrix<double>> C(9);

    Cr <<= 0.993701, 0.110304, -0.0197854, 0.280643,
          0.0815973, -0.833193, -0.546929, -0.255355,
          -0.0768135, 0.541869, -0.836945, 0.810979;

    C[0].resize(3,4);
    C[0] <<= 0.993479, 0.112002, -0.0213286, 0.287891,
            0.0822353, -0.83349, -0.54638, -0.255839,
            -0.0789729, 0.541063, -0.837266, 0.808608;

    C[1].resize(3,4);
    C[1] <<= 0.993199, 0.114383, -0.0217434, 0.295475,
            0.0840021, -0.833274, -0.546442, -0.25538,
            -0.0806218, 0.540899, -0.837215, 0.805906;

    C[2].resize(3,4);
    C[2] <<= 0.992928, 0.116793, -0.0213061, 0.301659,
            0.086304, -0.833328, -0.546001, -0.254563,
            -0.081524, 0.5403, -0.837514, 0.804653;

    C[3].resize(3,4);
    C[3] <<= 0.992643, 0.119107, -0.0217442, 0.309666,
            0.0880017, -0.833101, -0.546075, -0.254134,
            -0.0831565, 0.540144, -0.837454, 0.802222;

    C[4].resize(3,4);
    C[4] <<= 0.992429, 0.121049, -0.0208028, 0.314892,
            0.0901911, -0.833197, -0.545571, -0.253009,
            -0.0833736, 0.539564, -0.837806, 0.801559;

    C[5].resize(3,4);
    C[5] <<= 0.992226, 0.122575, -0.0215154, 0.32067,
            0.0911582, -0.833552, -0.544869, -0.254142,
            -0.0847215, 0.538672, -0.838245, 0.799812;

    C[6].resize(3,4);
    C[6] <<= 0.992003, 0.124427, -0.0211509, 0.325942,
            0.0930933, -0.834508, -0.543074, -0.254865,
            -0.0852237, 0.536762, -0.839418, 0.799037;

    C[7].resize(3,4);
    C[7] <<= 0.991867, 0.125492, -0.021234, 0.332029,
            0.0938678, -0.833933, -0.543824, -0.252767,
            -0.0859533, 0.537408, -0.838931, 0.797979;

    C[8].resize(3,4);
    C[8] <<= 0.991515, 0.128087, -0.0221943, 0.33934,
            0.095507, -0.833589, -0.544067, -0.250995,
            -0.0881887, 0.53733, -0.838748, 0.796756;

    QImage im;
    im = refim.convertToFormat(QImage::Format_Grayscale8);
    ps.HostRef8u.setSize(im.width(), im.height(), 0);
    ps.HostRef8u.data = im.bits();
    ps.CmatrixToRT(Cr, ps.HostRef8u.R, ps.HostRef8u.t);

    QString src;
    QImage isrc[9];
    ps.HostSrc8u.resize(9);
    for (int i = 1; i < 10; i++){
        src = QDir::currentPath();
        src += loc;
        src += QString::number(i);
        src += ".png";
        isrc[i-1].load(src);
        isrc[i-1] = isrc[i-1].convertToFormat(QImage::Format_Grayscale8);
        ps.HostSrc8u[i-1].setSize(isrc[i-1].width(), isrc[i-1].height(), 0);
        ps.HostSrc8u[i-1].data = isrc[i-1].bits();
        ps.CmatrixToRT(C[i-1], ps.HostSrc8u[i-1].R, ps.HostSrc8u[i-1].t);
    }

    ps.Convert8uTo32f(argc, argv);

}

PCLViewer::~PCLViewer ()
{
    delete[] argv;
    delete ui;
}

void PCLViewer::on_pushButton_pressed()
{
    depth = ps.RunAlgorithm(argc, argv);

    // Setup the cloud pointer
    cloud.reset (new PointCloudT);
    // The number of points in the cloud
    cloud->points.resize (depth.width * depth.height);

    // The default color
    red   = 128;
    green = 128;
    blue  = 128;

    // Fill the cloud with some points
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        cloud->points[i].x = i % depth.width;
        cloud->points[i].y = floor(i / depth.width);
        cloud->points[i].z = depth.data[i];

        cloud->points[i].r = red;
        cloud->points[i].g = green;
        cloud->points[i].b = blue;
    }

    viewer->addPointCloud (cloud, "cloud");
    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();
}
