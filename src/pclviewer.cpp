#define _SCL_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "pclviewer.h"
#include "ui_pclviewer.h"
#include <iostream>
#include <QDir>
#include <QByteArray>
#include <QBuffer>
#include <boost/numeric/ublas/assignment.hpp>
#include <QPixmap>
#include <QColor>
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QStringList>
#include <QMessageBox>
#include <cmath>
#include <pcl/io/pcd_io.h>

PCLViewer::PCLViewer (int argc, char **argv, QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::PCLViewer),
    scene (new QGraphicsScene()),
    depthscene (new QGraphicsScene()),
    dendepthsc (new QGraphicsScene()),
    tgvscene (new QGraphicsScene()),
    ps(argc, argv)
{

    QChar alpha = QChar(0xb1, 0x03);
    QChar sigma(0x03C3), tau(0x03C4), beta(0x03B2), gamma(0x03B3), theta(0x03B8);

    ui->setupUi (this);
    this->setWindowTitle ("PCL viewer");

    // Setup the cloud pointer
    cloud.reset (new PointCloudT);

    // Set up the QVTK window
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
    viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
    ui->qvtkWidget->update ();

    // Setup the cloud pointer
    clouddenoised.reset (new PointCloudT);

    // Set up the QVTK window
    viewerdenoised.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkDenoised->SetRenderWindow (viewerdenoised->getRenderWindow ());
    viewerdenoised->setupInteractor (ui->qvtkDenoised->GetInteractor (), ui->qvtkDenoised->GetRenderWindow ());
    ui->qvtkDenoised->update ();

    // Setup the cloud pointer
    tgvcloud.reset (new PointCloudT);

    // Set up the QVTK window
    tgvviewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtktgv->SetRenderWindow (tgvviewer->getRenderWindow ());
    tgvviewer->setupInteractor (ui->qvtktgv->GetInteractor (), ui->qvtktgv->GetRenderWindow ());
    ui->qvtktgv->update ();

    connect(ui->pSlider, SIGNAL(valueChanged(int)), this, SLOT(pSliderValueChanged(int)));

    viewer->addPointCloud (cloud, "cloud");
    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();

    viewerdenoised->addPointCloud (clouddenoised, "cloud");
    on_pSlider2_valueChanged (2);
    viewerdenoised->resetCamera ();
    ui->qvtkDenoised->update ();

    tgvviewer->addPointCloud (tgvcloud, "cloud");
    ui->tgv_psize->setValue(2);
    tgvviewer->resetCamera ();
    ui->qvtktgv->update ();

    ui->depthview->setScene(depthscene);

    ui->imNumber->setValue(ps.getNumberofImages());
    ui->winSize->setValue((ps.getWindowSize()));
    ui->nccThresh->setValue(ps.getNCCthreshold());
    ui->stdThresh->setValue(ps.getSTDthreshold());
    ui->numberPlanes->setValue(ps.getNumberofPlanes());
    ui->zNear->setValue(ps.getZnear());
    ui->zFar->setValue(ps.getZfar());
    ui->stdThresh->setMaximum(255 * 1.f);

    ui->tvl1_lambda_label->setText( trUtf8( "\xce\xbb" ) );
    ui->tvl1_sigma_label->setText(sigma);
    ui->tvl1_tau_label->setText(tau);
    ui->tvl1_theta_label->setText(theta);
    ui->tvl1_beta_label->setText(beta);
    ui->tvl1_gamma_label->setText(gamma);
    ui->lambda->setValue(DEFAULT_TVL1_LAMBDA);
    ui->nIters->setValue(DEFAULT_TVL1_ITERATIONS);
    ui->tvl1_sigma->setValue(DEFAULT_TVL1_SIGMA);
    ui->tvl1_tau->setValue(DEFAULT_TVL1_TAU);
    ui->tvl1_theta->setValue(DEFAULT_TVL1_THETA);
    ui->tvl1_beta->setValue(DEFAULT_TVL1_BETA);
    ui->tvl1_gamma->setValue(DEFAULT_TVL1_GAMMA);

    ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
    ui->threadsx->setMaximum(ui->maxthreads->value());
    ui->threadsy->setMaximum(ui->maxthreads->value());
    dim3 t = ps.getThreadsPerBlock();
    ui->threadsx->setValue(t.x);
    ui->threadsy->setValue(t.y);

    QString a0 = alpha, a1 = alpha;
    a0 += "<sub>";
    a0 += QString::number(0);
    a0 += "</sub>";
    a1 += "<sub>";
    a1 += QString::number(1);
    a1 += "</sub>";
    ui->alpha0_label->setText(a0);
    ui->alpha1_label->setText(a1);
    ui->tgv_lambda_label->setText(trUtf8( "\xce\xbb" ));
    ui->tgv_sigma_label->setText(sigma);
    ui->tgv_tau_label->setText(tau);
    ui->tgv_beta_label->setText(beta);
    ui->tgv_gamma_label->setText(gamma);

    ui->tgv_alpha0->setValue(DEFAULT_TGV_ALPHA0);
    ui->tgv_alpha1->setValue(DEFAULT_TGV_ALPHA1);
    ui->tgv_lambda->setValue(DEFAULT_TGV_LAMBDA);
    ui->tgv_niters->setValue(DEFAULT_TGV_NITERS);
    ui->tgv_warps->setValue(DEFAULT_TGV_NWARPS);
    ui->tgv_sigma->setValue(DEFAULT_TGV_SIGMA);
    ui->tgv_tau->setValue(DEFAULT_TGV_TAU);
    ui->tgv_beta->setValue(DEFAULT_TGV_BETA);
    ui->tgv_gamma->setValue(DEFAULT_TGV_GAMMA);

    ui->altmethod->setChecked(ps.getAlternativeRelativeMatrixMethod());

    LoadImages();
}

void
PCLViewer::pSliderValueChanged (int value)
{
    ui->cloudSize->setValue(value);
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
    int w = refim.width(), h = refim.height();

    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    double K[3][3] = {
        {0.709874*640, (1-0.977786)*640, 0.493648*640},
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

    //refgray = refim.convertToFormat(QImage::Format_Grayscale8);
    //ps.HostRef8u.CopyFrom(refgray.constBits(), refgray.bytesPerLine(), refgray.width(), refgray.height());
    ps.HostRef.setSize(w, h);
    rgb2gray<float>(ps.HostRef.data, refim);
    ps.CmatrixToRT(Cr, ps.HostRef.R, ps.HostRef.t);

    QString src;
	sources.resize(9);
    ps.HostSrc.resize(9);
    for (int i = 0; i < 9; i++){
        src = QDir::currentPath();
        src += loc;
        src += QString::number(i + 1);
        src += ".png";
        sources[i].load(src);
        //sources[i] = sources[i].convertToFormat(QImage::Format_Grayscale8);
        //ps.HostSrc8u[i].CopyFrom(sources[i].constBits(), sources[i].bytesPerLine(), sources[i].width(), sources[i].height());
        ps.HostSrc[i].setSize(w,h);
        rgb2gray<float>(ps.HostSrc[i].data, sources[i]);
        ps.CmatrixToRT(C[i], ps.HostSrc[i].R, ps.HostSrc[i].t);
    }

    //ps.Convert8uTo32f(argc, argv);
}

PCLViewer::~PCLViewer ()
{
    delete[] argv;
    delete ui;
}

void PCLViewer::on_pushButton_pressed()
{
    if (ps.RunAlgorithm(argc, argv)){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        depth = ps.getDepthmap();
        depth8u = ps.getDepthmap8u();
        if (refchanged) {
            if ((cloud->height != depth8u->height) || (cloud->width != depth8u->width))
            {
                // The number of points in the cloud
                cloud->points.resize(depth8u->width * depth8u->height);
                cloud->width = depth8u->width;
                cloud->height = depth8u->height;
            }
        }

        QColor c;
        int  i;
        double k[3][3];
        ps.getInverseK(k);
        double z;

        // Fill the cloud with some points
        for (size_t x = 0; x < depth8u->width; ++x)
            for (size_t y = 0; y < depth8u->height; ++y)
            {

                i = x + y * depth8u->width;
                z = depth->data[i];

                cloud->points[i].z = -z;
                cloud->points[i].x = z * (k[0][0] * x + k[0][1] * y + k[0][2]);
                cloud->points[i].y = z * (k[1][0] * x + k[1][1] * y + k[1][2]);

                if (refchanged)
                {
                    c = refim.pixel(x, y);
                    cloud->points[i].r = c.red();
                    cloud->points[i].g = c.green();
                    cloud->points[i].b = c.blue();
                }
                //depth8u->data[i] = (uchar)d;
            }

        QImage img((const uchar *)depth8u->data, depth8u->width, depth8u->height, QImage::Format_Indexed8);

        depthim = QPixmap::fromImage(img);
        depthscene->addPixmap(depthim);
        depthscene->setSceneRect(depthim.rect());
        ui->depthview->setScene(depthscene);

        viewer->updatePointCloud(cloud, "cloud");
        if (refchanged) viewer->resetCamera();
        ui->qvtkWidget->update();
        refchanged = false;
    }
}

void PCLViewer::on_imNumber_valueChanged(int arg1)
{
    ps.setNumberofImages(arg1);
}

void PCLViewer::on_winSize_valueChanged(int arg1)
{
    // Make sure arg1 is odd
    arg1 = 2 * (arg1 / 2) + 1;
    ui->winSize->setValue(arg1);
    ps.setWindowSize(arg1);
}

void PCLViewer::on_numberPlanes_valueChanged(int arg1)
{
    ps.setNumberofPlanes(arg1);
}

void PCLViewer::on_zNear_valueChanged(double arg1)
{
    ps.setZnear(arg1);
}

void PCLViewer::on_zFar_valueChanged(double arg1)
{
    ps.setZfar(arg1);
}

void PCLViewer::on_stdThresh_valueChanged(double arg1)
{
    ps.setSTDthreshold(arg1);
}

void PCLViewer::on_nccThresh_valueChanged(double arg1)
{
    ps.setNCCthreshold(arg1);
}

void PCLViewer::on_pSlider2_valueChanged(int value)
{
    ui->cloudSize2->setValue(value);
    viewerdenoised->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtkDenoised->update ();
}

void PCLViewer::on_denoiseBtn_clicked()
{
//    if (ps.Denoise(ui->nIters->value(), ui->lambda->value())){
    if (ps.CudaDenoise(argc, argv, ui->nIters->value(), ui->lambda->value(), ui->tvl1_tau->value(),
                       ui->tvl1_sigma->value(), ui->tvl1_theta->value(), ui->tvl1_beta->value(), ui->tvl1_gamma->value())){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        dendepth = ps.getDepthmapDenoised();
        ps.get3Dcoordinates(cx, cy, cz);
        dendepth8u = ps.getDepthmap8uDenoised();
        int size = clouddenoised->points.size();
        if (refchangedtvl1)
        {
//            if ((clouddenoised->height != dendepth8u->height) || (clouddenoised->width != dendepth8u->width))
//            {
//                // The number of points in the cloud
//                clouddenoised->points.resize(dendepth8u->width * dendepth8u->height);
//                clouddenoised->width = dendepth8u->width;
//                clouddenoised->height = dendepth8u->height;
//            }
            clouddenoised->points.resize(size + dendepth8u->width * dendepth8u->height);
            clouddenoised->width = dendepth8u->width;
            clouddenoised->height += dendepth8u->height;
        }

        size = clouddenoised->points.size() - dendepth8u->width * dendepth8u->height;

        QColor c;
        int  i;

//        double k[3][3];
//        ps.getInverseK(k);
//        double z;

        // Fill the cloud with some points
        for (size_t x = 0; x < dendepth8u->width; ++x)
            for (size_t y = 0; y < dendepth8u->height; ++y)
            {

                i = x + y * dendepth8u->width;

//                z = dendepth->data[i];
//                clouddenoised->points[i].z = -z;
//                clouddenoised->points[i].x = z * (k[0][0] * x + k[0][1] * y + k[0][2]);
//                clouddenoised->points[i].y = z * (k[1][0] * x + k[1][1] * y + k[1][2]);
                if (cz->data[i] != -9.f)
                {
                    clouddenoised->points[i + size].z = -cz->data[i];
                    clouddenoised->points[i + size].x = cx->data[i];
                    clouddenoised->points[i + size].y = cy->data[i];
                }

                if (refchangedtvl1)
                {
                    c = refim.pixel(x, y);
                    clouddenoised->points[i + size].r = c.red();
                    clouddenoised->points[i + size].g = c.green();
                    clouddenoised->points[i + size].b = c.blue();
                }
            }

        QImage img((const uchar *)dendepth8u->data, dendepth8u->width, dendepth8u->height, QImage::Format_Indexed8);

        dendepthim = QPixmap::fromImage(img);
        dendepthsc->addPixmap(dendepthim);
        dendepthsc->setSceneRect(dendepthim.rect());
        ui->denview->setScene(dendepthsc);

        viewerdenoised->updatePointCloud(clouddenoised, "cloud");
        if (refchangedtvl1) viewerdenoised->resetCamera();
        ui->qvtkDenoised->update();
        refchangedtvl1 = false;
    }
}

void PCLViewer::on_threadsx_valueChanged(int arg1)
{
    ps.setBlockXdim(arg1);
    ui->threadsx->setValue(arg1);
}

void PCLViewer::on_threadsy_valueChanged(int arg1)
{
    ps.setBlockYdim(arg1);
    ui->threadsy->setValue(arg1);
}

void PCLViewer::on_tgv_button_pressed()
{
    if (ps.TGV(argc, argv, ui->tgv_niters->value(), ui->tgv_warps->value(), ui->tgv_lambda->value(),
               ui->tgv_alpha0->value(), ui->tgv_alpha1->value(), ui->tgv_tau->value(), ui->tgv_sigma->value(),
               ui->tgv_beta->value(), ui->tgv_gamma->value())){
        ui->maxthreads->setValue(ps.getMaxThreadsPerBlock());
        tgvdepth = ps.getDepthmapTGV();
        tgvdepth8u = ps.getDepthmap8uTGV();
        if (refchangedtgv)
        {
            // The number of points in the cloud
            if ((tgvcloud->height != tgvdepth8u->height) || (tgvcloud->width != tgvdepth8u->width))
            {
                tgvcloud->points.resize(tgvdepth8u->width * tgvdepth8u->height);
                tgvcloud->width = tgvdepth8u->width;
                tgvcloud->height = tgvdepth8u->height;
            }
        }

        QColor c;
        int  i;
        double k[3][3];
        ps.getInverseK(k);
        double z, zn, zf;
        zn = ps.getZnear();
        zf = ps.getZfar();

        // Fill the cloud with some points
        for (size_t x = 0; x < tgvdepth8u->width; ++x)
            for (size_t y = 0; y < tgvdepth8u->height; ++y)
            {

                i = x + y * tgvdepth8u->width;

                z = ((float)tgvdepth8u->data[i] / 255.f * (zf - zn));
                tgvcloud->points[i].z = -z;
                tgvcloud->points[i].x = z * (k[0][0] * x + k[0][1] * y + k[0][2]);
                tgvcloud->points[i].y = z * (k[1][0] * x + k[1][1] * y + k[1][2]);

                if (refchangedtgv)
                {
                    c = refim.pixel(x, y);
                    tgvcloud->points[i].r = c.red();
                    tgvcloud->points[i].g = c.green();
                    tgvcloud->points[i].b = c.blue();
                }
            }

        QImage img((const uchar *)tgvdepth8u->data, tgvdepth8u->width, tgvdepth8u->height, QImage::Format_Indexed8);

        tgvdepthim = QPixmap::fromImage(img);
        tgvscene->addPixmap(tgvdepthim);
        tgvscene->setSceneRect(tgvdepthim.rect());
        ui->tgvview->setScene(tgvscene);

        tgvviewer->updatePointCloud(tgvcloud, "cloud");
        if (refchangedtgv) tgvviewer->resetCamera();
        ui->qvtktgv->update();
        refchangedtgv = false;
    }
}

void PCLViewer::on_tgv_psize_valueChanged(int value)
{
    ui->tgv_psizebox->setValue(value);
    tgvviewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    ui->qvtktgv->update ();
}

void PCLViewer::on_imagePathButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                 "/home",
                                                 QFileDialog::ShowDirsOnly
                                                 | QFileDialog::DontResolveSymlinks);
    ui->imagePath->setText(dir);
}

void PCLViewer::on_imageNameButton_clicked()
{
    // Get full image file name
    QString loc = ui->imagePath->text();
    if (loc.isEmpty()) loc = "/home";
    QString name = QFileDialog::getOpenFileName(this, tr("Select Reference Image"),
                                                loc,
                                                tr("Images (*.png *.xpm *.jpg *.jpeg *.bmp *.dds *.mng *.tga *.tiff)"));

    // Find the last forward slash, meaning end directory path
    int i = name.lastIndexOf('/');

    // Split full file name into name and path
    QString n = name, path = name;
    if (i != -1) {
        path.truncate(i);
        n.remove(0, i + 1);
    }

    // Split file name into name and format
    QString format = n;
    i = n.lastIndexOf('.');
    if (i != 1) {
        format.remove(0, i + 1);
        n.truncate(i);
    }

    // Going from the end count the number of digits and selected image number
    int digits = 0, ref = 0;
    QChar s = n.at(n.size() - 1);
    while (s.isDigit()){
        ref += pow(10, digits) * s.digitValue();
        digits++;
        n.truncate(n.size() - 1);
        s = n.at(n.size() - 1);
    }

    // Set appropriate boxes
    ui->imagePath->setText(path);
    ui->imageName->setText(n);
    ui->imageFormat->setText(format);
    ui->imageDigits->setValue(digits);
    ui->refNumber->setValue(ref);
}

void PCLViewer::on_altmethod_toggled(bool checked)
{
    ps.setAlternativeRelativeMatrixMethod(checked);
}

void PCLViewer::on_loadfromsrc_clicked()
{
    refchanged = true;
    refchangedtvl1 = true;
    refchangedtgv = true;
    LoadImages();
}

void PCLViewer::on_loadfromdir_clicked()
{
    refchanged = true;
    refchangedtvl1 = true;
    refchangedtgv = true;
    QString impos;
    QString imname = ImageName(ui->refNumber->value(), impos);

    ublas::matrix<double> cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, K, R, t;
    double cam_angle;

    if (!getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)) return;
    getcamK(K, cam_dir, cam_up, cam_right);

    std::cout << "cam_dir = " << cam_dir << std::endl;
    ps.setK(K);
    computeRT(R, t, cam_dir, cam_pos, cam_up);

    refim.load(imname);
    int w = refim.width(), h = refim.height();
    ps.HostRef.setSize(w, h);

    image = QPixmap::fromImage(refim);
    scene->addPixmap(image);
    scene->setSceneRect(image.rect());
    ui->refView->setScene(scene);

    ps.HostRef.R = R;
    ps.HostRef.t = t;
    rgb2gray<float>(ps.HostRef.data, refim);
    std::cout << "K = " << K << std::endl;
    std::cout << "Rref = " << R << "\ntref = " << t << std::endl << std::endl;

    int nsrc = ui->imNumber->value() - 1;

    ps.HostSrc.resize(0);
    QImage src;
    int half = (nsrc + 1) / 2;
    int offset;

    for (int i = 0; i < nsrc; i++){
        if (i < half) offset = i + 1;
        else offset = half - i - 1;
        imname = ImageName(ui->refNumber->value() + offset, impos);
        if (getcamParameters(impos, cam_pos, cam_dir, cam_up, cam_lookat,cam_sky, cam_right, cam_fpoint, cam_angle)){
            ps.HostSrc.resize(ps.HostSrc.size() + 1);
            computeRT(R, t, cam_dir, cam_pos, cam_up);
            src.load(imname);
            ps.HostSrc.back().setSize(w,h);
            ps.HostSrc.back().R = R;
            ps.HostSrc.back().t = t;
            rgb2gray<float>(ps.HostSrc.back().data, src);
            std::cout << "i = "<< i << "\nRsens = " << R << "\ntsens = " << t << std::endl << std::endl;
        }
    }

}

QString PCLViewer::ImageName(int number, QString &imagePos)
{
    QString name = ui->imagePath->text();
    name += '/';
    name += ui->imageName->text();
    int n;
    for (int i = 0; i < ui->imageDigits->value(); i++) {
        n = number / (int)pow(10, ui->imageDigits->value() - i - 1);
        n = n % 10;
        name += QString::number(n);
    }
    name += '.';
    imagePos = name;
    name += ui->imageFormat->text();
    imagePos += "txt";
    return name;
}

void PCLViewer::getcamK(ublas::matrix<double> & K, const ublas::matrix<double> &cam_dir,
                        const ublas::matrix<double> &cam_up, const ublas::matrix<double> &cam_right)
{
    double focal = sqrt(pow(cam_dir(0,0),2) + pow(cam_dir(1,0),2) + pow(cam_dir(2,0),2));
    double aspect = sqrt(pow(cam_right(0,0),2) + pow(cam_right(1,0),2) + pow(cam_right(2,0),2));
    double angle = 2 * atan(aspect / 2 / focal);
    aspect = aspect / sqrt(pow(cam_up(0,0),2) + pow(cam_up(1,0),2) + pow(cam_up(2,0),2));

    // height and width
    int M = 480, N = 640;

    int width = N, height = M;

    // pixel size
    double psx = 2*focal*tan(0.5*angle)/N ;
    double psy = 2*focal*tan(0.5*angle)/aspect/M ;

    psx   = psx / focal;
    psy   = psy / focal;

    double Ox = (width+1)*0.5;
    double Oy = (height+1)*0.5;

    K.resize(3,3);

    K <<=   1.f/psx, 0.f, Ox,
            0.f, -1.f/psy, Oy,
            0.f, 0.f, 1.f;
}

void PCLViewer::computeRT(ublas::matrix<double> & R, ublas::matrix<double> & t, const ublas::matrix<double> &cam_dir,
                          const ublas::matrix<double> &cam_pos, const ublas::matrix<double> &cam_up)
{
    ublas::matrix<double> x, y, z;

    z = cam_dir / sqrt(pow(cam_dir(0,0),2) + pow(cam_dir(1,0),2) + pow(cam_dir(2,0),2));

    x = cross(cam_up, z);
    x = x / sqrt(pow(x(0,0),2) + pow(x(1,0),2) + pow(x(2,0),2));

    y = cross(z, x);

    R.resize(3,3);
    R <<=   x(0,0), y(0,0), z(0,0),
            x(1,0), y(1,0), z(1,0),
            x(2,0), y(2,0), z(2,0);

    t = cam_pos;

}

bool PCLViewer::getcamParameters(QString filename, ublas::matrix<double> & cam_pos, ublas::matrix<double> & cam_dir,
                                 ublas::matrix<double> & cam_up, ublas::matrix<double> & cam_lookat,
                                 ublas::matrix<double> & cam_sky, ublas::matrix<double> & cam_right,
                                 ublas::matrix<double> & cam_fpoint, double & cam_angle)
{
    cam_pos.resize(3,1);
    cam_dir.resize(3,1);
    cam_up.resize(3,1);
    cam_lookat.resize(3,1);
    cam_sky.resize(3,1);
    cam_right.resize(3,1);
    cam_fpoint.resize(3,1);

    QFile file(filename);
    if(!file.open(QIODevice::ReadOnly)) {
        QMessageBox::information(0, "Error reading file", file.errorString());
        return false;
    }

    QTextStream in(&file);
    int first, last;

    while(!in.atEnd()) {
        QString line = in.readLine();
        QString numbers = line;
        QStringList n;

        first = line.lastIndexOf('[');
        last = line.lastIndexOf(']');

        if (last != -1) numbers.truncate(last);
        if (first != -1) numbers.remove(0, first + 1);
        if ((first != -1) && (last != -1)) n = numbers.split(',', QString::SkipEmptyParts);

        if (line.startsWith(CAM_POS)){
            cam_pos <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_DIR)){
            cam_dir <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_UP)){
            cam_up <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_LOOKAT)){
            cam_lookat <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_SKY)){
            cam_sky <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_RIGHT)){
            cam_right <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_FPOINT)){
            cam_fpoint <<= n.at(0).trimmed().toDouble(), n.at(1).trimmed().toDouble(), n.at(2).trimmed().toDouble();
        }

        if (line.startsWith(CAM_ANGLE)){
            first = line.lastIndexOf('=');
            last = line.lastIndexOf(';');
            if (last != -1) numbers.truncate(last);
            if (first != -1) numbers.remove(0, first + 1);
            if ((first != -1) && (last != -1)) n = numbers.split(',', QString::SkipEmptyParts);
            cam_angle = n.at(0).trimmed().toDouble();
        }

    }

    file.close();
    return true;
}

ublas::matrix<double> & PCLViewer::cross(const ublas::matrix<double> & A, const ublas::matrix<double> & B)
{
    ublas::matrix<double> x = A, y = B;
    ublas::matrix<double> * result = new ublas::matrix<double>(3,1);
    int xdim = 3, ydim = 3;
    if (x.size1() == 1) x = trans(x);
    if (y.size1() == 1) y = trans(y);
    if (x.size1() == 2) {
        xdim = 2;
        x.resize(3, 1);
        x(2,0) = 0.f;
    }
    else if (x.size1() != 3) {
        std::cerr << "Matrix dimensions must be 3 by 1 or 2 by 1 only" << std::endl;
        return *result;
    }
    if (y.size1() == 2) {
        ydim = 2;
        y.resize(3, 1);
        y(2,0) = 0.f;
    }
    else if (y.size1() != 3) {
        std::cerr << "Matrix dimensions must be 3 by 1 or 2 by 1 only" << std::endl;
        return *result;
    }

    *result <<=  x(1,0) * y(2,0) - x(2,0) * y(1,0),
                -(x(0,0) * y(2,0) - x(2,0) * y(0,0)),
                x(0,0) * y(1,0) - x(1,0) * y(0,0);

    return *result;
}

template<typename T>
void PCLViewer::rgb2gray(T * data, const QImage & img)
{
    int w = img.width(), h = img.height();
    QColor c;

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){

            c = img.pixel(x, y);
            data[x + y*w] = T(RGB2GRAY_WEIGHT_RED * c.red() +
                              RGB2GRAY_WEIGHT_BLUE * c.blue() +
                              RGB2GRAY_WEIGHT_GREEN * c.green());
        }
    }
}

void PCLViewer::on_save_clicked()
{
    QFile file;

    if (!refim.isNull())
    {
        file.setFileName("reference.png");
        file.open(QIODevice::WriteOnly);
        refim.save(&file, "PNG");
        file.close();
    }

    if (!depthim.isNull())
    {
        file.setFileName("planesweep.png");
        file.open(QIODevice::WriteOnly);
        depthim.save(&file, "PNG");
        file.close();
    }

    if (!dendepthim.isNull())
    {
        file.setFileName("planesweep_tvl1.png");
        file.open(QIODevice::WriteOnly);
        dendepthim.save(&file, "PNG");
        file.close();
    }

    if (!tgvdepthim.isNull())
    {
        file.setFileName("tgv.png");
        file.open(QIODevice::WriteOnly);
        tgvdepthim.save(&file, "PNG");
        file.close();
    }

    try {
        pcl::io::savePCDFileASCII("planesweep.pcd", *cloud);
        pcl::io::savePCDFileASCII("planesweep_tvl1.pcd", *clouddenoised);
        pcl::io::savePCDFileASCII("tgv.pcd", *tgvcloud);
    }
    catch (pcl::IOException & excep){
        std::cerr << "Error occured while saving PCD:\n" << excep.detailedMessage() << std::endl;
        return;
    }
}

void PCLViewer::on_nIters_valueChanged(int arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_lambda_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_tau_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_sigma_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_theta_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_beta_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_tvl1_gamma_valueChanged(double arg1)
{
    if (ui->tvl1_rtupdate_box->isChecked()) on_denoiseBtn_clicked();
}

void PCLViewer::on_reconstruct_button_clicked()
{
    for (int i = 0; i < 25; i++){
        ui->refNumber->setValue(ui->refNumber->value() + 20);
        on_loadfromdir_clicked();
        on_pushButton_pressed();
        on_denoiseBtn_clicked();
    }
}
