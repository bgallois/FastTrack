#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QTimer>
#include <QFile>
#include <QFileInfo>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <functions.h>



using namespace cv;
using namespace std;


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    // Setup the ui
    ui->setupUi(this);
    setWindowState(Qt::WindowMaximized);
    setWindowTitle("Fishy Tracking");

    // Setup style
    QFile stylesheet(":/darkTheme.qss");

    if(stylesheet.open(QIODevice::ReadOnly | QIODevice::Text)) { // Read the theme file
        qApp->setStyleSheet(stylesheet.readAll());
        stylesheet.close();
    }

    else{ // If the theme is not found print an error (for dev purpose)
        cout << "Theme not found" << '\n';
    }


    // Welcome message
    welcomeBox = new QMessageBox;
    welcomeBox ->setText("Check new versions at https://bgallois.github.io/FishyTracking/.\n © Benjamin GALLOIS benjamin.gallois@upmc.fr. \n Subscribe to the mailing list to be kept informed of new releases and new features: http://benjamin-gallois.fr/fishyTracking.html");


    // Default parameters reading
    QStringList defParameters;
    QFile file("conf.txt");

    if(QFileInfo("conf.txt").exists()){ // Existing default parameters
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        QTextStream in(&file);

        while(!in.atEnd()) { // Create parameters file
            QString line = in.readLine();
            defParameters.append(line);
        }

        file.close();
    }

    else{ // Create a defaults files if not existing
        int i = 0;
        while(i<15) {
            QString line = "0";
            defParameters.append(line);
            i++;
        }
    }


    // Widgets layout

    QuitButton = new QPushButton("Quit", this);
    QObject::connect(QuitButton, SIGNAL(clicked()), qApp, SLOT(quit()));
    QObject::connect(qApp, SIGNAL(aboutToQuit()), welcomeBox, SLOT(exec()));


    ResetButton = new QPushButton("Reset", this);
    QObject::connect(ResetButton, SIGNAL(clicked()), this, SLOT(Reset()));


    GoButton = new QPushButton("Go", this);
    QObject::connect(GoButton, SIGNAL(clicked()), this, SLOT(Go()));

    PauseButton = new QPushButton("Pause", this);
    pause = true;

    DefaultButton = new QPushButton("Set as default", this);
    QObject::connect(DefaultButton, SIGNAL(clicked()), this, SLOT(Write()));

    ReplayButton = new QPushButton("Replay", this);
    QObject::connect(ReplayButton, SIGNAL(clicked()), this, SLOT(Replay()));
    QSizePolicy spRetain = ReplayButton->sizePolicy();
    spRetain.setRetainSizeWhenHidden(true);
    ReplayButton->setSizePolicy(spRetain);
    ReplayButton ->hide();


    normal = new QCheckBox("Normal", this);
    normal ->setChecked(1);
    normal ->isChecked();
    normal ->setToolTip(tr("Check this option to display the original image with trajectories of tracked objects."));

    binary = new QCheckBox("Binary", this);
    binary ->setToolTip(tr("Check this option to display the binary image. Useful to ajust the binary threshold parameter."));



    path = new QLabel(this);
    path->setText("Path:");
    path->adjustSize();
    path ->setToolTip(tr("Path to the folder that contains images. Have to add /*extension at the end."));


    pathField = new QLineEdit(this);
    pathField->setText(defParameters.at(0));
    pathField->adjustSize();
    QObject::connect(pathField, SIGNAL(textChanged(QString)), this, SLOT(checkPath(QString)));


    numLabel = new QLabel(this);
    numLabel->setText("Number of objects:");
    numLabel->adjustSize();
    numLabel->setToolTip(tr("Number of objects to track, can't be changed during the tracking phase."));


    numField = new QLineEdit(this);
    numField->setText(defParameters.at(1));


    maxArea = new QLabel(this);
    maxArea->setText("Maximal area:");
    maxArea->adjustSize();
    maxArea->setToolTip(tr("Maximal area in px of objects to track."));

    maxAreaField = new QLineEdit(this);
    maxAreaField->setText(defParameters.at(2));


    minArea = new QLabel(this);
    minArea->setText("Minimal area:");
    minArea->adjustSize();
    minArea->setToolTip(tr("Minimal area in px of objects to track."));


    minAreaField = new QLineEdit(this);
    minAreaField->setText(defParameters.at(3));


    thresh = new QLabel(this);
    thresh->setText("Binary threshold:");
    thresh->adjustSize();
    thresh->setToolTip(tr("Binary threshold, image[i,j] < thresh = 0, image[i,j] > thresh = 1."));


    threshField = new QLineEdit(this);
    threshField->setText(defParameters.at(9));
    threshField->adjustSize();


    length = new QLabel(this);
    length->setText("Max displacement:");
    length->adjustSize();
    length->setToolTip(tr("Maximal displacement possible of an object between two frames in px."));



    lengthField = new QLineEdit(this);
    lengthField->setText(defParameters.at(4));


    angle = new QLabel(this);
    angle->setText("Max angle:");
    angle->adjustSize();
    angle->setToolTip(tr("Maximal orientation change of an object between two frames in degree."));


    angleField = new QLineEdit(this);
    angleField->setText(defParameters.at(5));



    lo = new QLabel(this);
    lo->setText("Max occlusion:");
    lo->adjustSize();
    lo->setToolTip(tr("Maximum distance in px that an object can move when it is occulted by another object."));



    loField = new QLineEdit(this);
    loField->setText(defParameters.at(6));


    w = new QLabel(this);
    w->setText("Weight:");
    w->adjustSize();
    w->setToolTip(tr("Importance of orientation and distance for the tracking. Closer to 0 the tracking is more sensitive to the orientation of objects and closer to 1 is more sensitive to the distance."));



    wField = new QLineEdit(this);
    wField->setText(defParameters.at(7));


    nBack = new QLabel(this);
    nBack->setText("Background images:");
    nBack->adjustSize();
    nBack->setToolTip(tr("Number of images to compute the background."));



    nBackField = new QLineEdit(this);
    nBackField->setText(defParameters.at(8));


    save = new QLabel(this);
    save->setText("Save to:");
    save->adjustSize();
    save->setToolTip(tr("Path to a .txt file where the result of the tracking will be saved. Saved in the reference frame of the croped image."));


    saveField = new QLineEdit(this);
    saveField->setText(defParameters.at(10));
    QObject::connect(pathField, SIGNAL(textChanged(QString)), this, SLOT(setSavePath(QString)));


    x1ROI = new QLabel(this);
    x1ROI->setText("Top corner x position for the ROI:");
    x1ROI->adjustSize();
    x1ROI->setToolTip(tr("Horizontale position of the top right corner of the region of interest."));



    x1ROIField = new QLineEdit(this);
    x1ROIField->setText(defParameters.at(11));



    x2ROI = new QLabel(this);
    x2ROI->setText("Bottom corner x position for the ROI:");
    x2ROI->adjustSize();
    x2ROI->setToolTip(tr("Horizontale position of the bottom left corner of the region of interest. Set to 0 for autoset."));


    x2ROIField = new QLineEdit(this);
    x2ROIField->setText(defParameters.at(12));



    y1ROI = new QLabel(this);
    y1ROI->setText("Top corner y position for the ROI:");
    y1ROI->adjustSize();
    y1ROI->setToolTip(tr("Verticale position of the top right corner of the region of interest."));


    y1ROIField = new QLineEdit(this);
    y1ROIField->setText(defParameters.at(13));



    y2ROI = new QLabel(this);
    y2ROI->setText("Bottom corner y position for the ROI:");
    y2ROI->adjustSize();
    y2ROI->setToolTip(tr("Verticale position of the bottom left corner of the region of interest. Set to 0 for autoset."));



    y2ROIField = new QLineEdit(this);
    y2ROIField->setText(defParameters.at(14));


    arrow = new QLabel(this);
    arrow -> setText("Arrow size:");


    arrowField = new QLCDNumber(this);
    arrowField ->display(2);


    arrowSlider = new QSlider(Qt::Horizontal, this);
    arrowSlider -> setMinimum(2);

    QObject::connect(arrowSlider, SIGNAL(valueChanged(int)), arrowField, SLOT(display(int))) ;

    fps = new QLabel(this);
    fps -> setText("FPS:");
    fps -> hide();


    fpsField = new QLCDNumber(this);
    fpsField ->display(20);
    fpsField -> hide();


    fpsSlider = new QSlider(Qt::Horizontal, this);
    fpsSlider -> setMinimum(1);
    fpsSlider -> setMaximum(300);
    fpsSlider-> hide();

    QObject::connect(fpsSlider, SIGNAL(valueChanged(int)), fpsField, SLOT(display(int)));

    trackingSpotLabel = new QLabel(this);
    trackingSpotLabel ->setText("Spot to track:");

    trackingSpot = new QComboBox(this);
    trackingSpot ->addItem(tr("Head"));
    trackingSpot ->addItem(tr("Tail"));
    trackingSpot ->addItem(tr("Center of mass"));
    trackingSpot ->adjustSize();



    display = new QLabel(this);
    display->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);



    display2 = new QLabel(this);
    display2->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);



    progressBar = new QProgressBar(this);

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(path, 0, 0);
    layout->addWidget(numLabel, 1, 0);
    layout->addWidget(maxArea, 2, 0);
    layout->addWidget(minArea, 3, 0);
    layout->addWidget(thresh, 4, 0);
    layout->addWidget(pathField, 0, 1);
    layout->addWidget(numField, 1, 1);
    layout->addWidget(maxAreaField, 2, 1);
    layout->addWidget(minAreaField, 3, 1);
    layout->addWidget(threshField, 4, 1);

    layout->addWidget(length, 0, 3);
    layout->addWidget(angle, 1, 3);
    layout->addWidget(lo, 2, 3);
    layout->addWidget(w, 3, 3);
    layout->addWidget(nBack, 4, 3);
    layout->addWidget(lengthField, 0, 4);
    layout->addWidget(angleField, 1, 4);
    layout->addWidget(loField, 2, 4);
    layout->addWidget(wField, 3, 4);
    layout->addWidget(nBackField, 4, 4);

    layout->addWidget(save, 0, 6);
    layout->addWidget(x1ROI, 1, 6);
    layout->addWidget(x2ROI, 2, 6);
    layout->addWidget(y1ROI, 3, 6);
    layout->addWidget(y2ROI, 4, 6);
    layout->addWidget(saveField, 0, 7);
    layout->addWidget(x1ROIField, 1, 7);
    layout->addWidget(x2ROIField, 2, 7);
    layout->addWidget(y1ROIField, 3, 7);
    layout->addWidget(y2ROIField, 4, 7);

    layout->addWidget(arrow, 0, 9);
    layout->addWidget(arrowField, 0, 10);
    layout->addWidget(arrowSlider, 1, 9);

    layout->addWidget(trackingSpotLabel, 2, 9);
    layout->addWidget(trackingSpot, 3, 9);

    layout->addWidget(fps, 4, 9);
    layout->addWidget(fpsField, 4, 10);
    layout->addWidget(fpsSlider, 5, 9);

    layout->addWidget(GoButton, 5, 0);
    layout->addWidget(PauseButton, 5, 1);
    layout->addWidget(ReplayButton, 5, 2);
    layout->addWidget(DefaultButton, 5, 3);
    layout->addWidget(QuitButton, 5, 4);
    layout->addWidget(ResetButton, 5, 5);

    layout->addWidget(normal, 5, 6);
    layout->addWidget(binary, 6, 6);

    layout->addWidget(progressBar, 7, 0, 1, 10);

    layout->addWidget(display, 8, 0, 5, 5);
    layout->addWidget(display2, 8, 6, 5, 5);


    central = new QWidget(this);
    central->setObjectName("central");
    central->setLayout(layout);
    setCentralWidget(central);




    // Initialization

    im = 0; // Internal counter
    timer = new QTimer(this); // Tracking timer
    timer ->setInterval(1);
    connect(timer, SIGNAL(timeout()), this, SLOT(Go())); // Tracking
    connect(this, SIGNAL(grabFrame(Mat, UMat)), this, SLOT(Display(Mat, UMat))); // Displaying

    imr = 0; // Internal counter for the replay
    timerReplay = new QTimer(this); 
    connect(timerReplay, SIGNAL(timeout()), this, SLOT(Replay())); // Replaying


    QObject::connect(PauseButton, SIGNAL(clicked()), this, SLOT(PlayPause()));

}


/**
    * @PlayPause: custom play/pause button that allow to stop restart the tracking, also in case of error.
*/
void MainWindow::PlayPause(){
    if (pause){ // Pause button pressed
        PauseButton ->setText("Play");
        timer ->stop();
        pause = false;
    }

    else if (pause == false){ // Play button pressed
        PauseButton ->setText("Pause");
        x2ROIField->setStyleSheet("background-color: black;");
        y2ROIField->setStyleSheet("background-color: black;");
        timer ->start();
        pause = true;
    }
}



/**
    * @UpdateParameters: take the parameters from fields, cast them to right type.
*/
void MainWindow::UpdateParameters(){

    spot = trackingSpot->currentIndex();
    arrowSize = arrowField-> value();
    MAXAREA = maxAreaField->text().toInt(); //9000; //minimal area of detected objects
    MINAREA = minAreaField->text().toInt(); //3000; //maximal area of detected objects
    LENGTH = lengthField->text().toInt(); //maximal distance covered between two frames
    ANGLE = (angleField->text().toDouble()*M_PI)/180; //maximal angle between two frames
    WEIGHT = wField->text().toDouble();// weight of the orientation/angle in the cost function, 0 = orientation, 1 = distance
    LO = loField->text().toInt(); // maximum occlusion distance
    threshValue = threshField->text().toInt();
    x1 = x1ROIField->text().toInt();
    x2 = x2ROIField->text().toInt();
    y1 = y1ROIField->text().toInt();
    y2 = y2ROIField->text().toInt();

}


/**
    * @Go: Main function for the tracking, take one image and track the object inside.
*/
void MainWindow::Go(){

        try{

        connect(trackingSpot, SIGNAL(currentIndexChanged(int)), this, SLOT(UpdateParameters()));
        connect(arrowSlider, SIGNAL(valueChanged(int)), this, SLOT(UpdateParameters()));
        connect(maxAreaField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(minAreaField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(lengthField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(angleField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(wField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(loField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(x1ROIField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(y1ROIField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(x2ROIField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(y2ROIField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));
        connect(threshField, SIGNAL(editingFinished()), this, SLOT(UpdateParameters()));



        // Widgets
        pathField->setDisabled(true);
        numField->setDisabled(true);
        nBackField->setDisabled(true);
        saveField->setDisabled(true);
        
        



        if(im == 0){ // Initialization


            timer->start();
            UpdateParameters();
            string folder = pathField->text().toStdString();
            nBackground = nBackField->text().toInt();
            NUMBER = numField->text().toInt(); //number of objects to track

            try{
                glob(folder, files, false);
                checkPath(pathField->text());
                UpdateParameters();
            }
            catch (...){
                timer->stop();
                PauseButton ->setText("Play");
                pause = false;
                QMessageBox pathError;
                pathError.setText("No files found");
                pathError.exec();
            }

            sort(files.begin(), files.end());
            a = files.begin();
            string name = *a;
            progressBar ->setRange(0, files.size());
            background = BackgroundExtraction(files, nBackground);
            //double concentration = Concentration(files);
            vector<vector<Point> > tmp(NUMBER, vector<Point>());
            memory = tmp;
            colorMap = Color(NUMBER);
            savePath = saveField->text().toStdString();

            ROI = AutoROI(background);
            
            



            for(unsigned int ini = 0; ini < files.size()*NUMBER; ini++){
                internalSaving.push_back(Point3f(0., 0., 0.));
            }
        }

        
        string name = *a;
        savefile.open(savePath);

        //Rect ROI(x1, y1, x2 - x1, y2 - y1);
        imread(name, IMREAD_GRAYSCALE).copyTo(cameraFrame);
        string metadata = Metadata(name);

        visu = cameraFrame.getMat(ACCESS_FAST).clone();

        subtract(background, cameraFrame, cameraFrame);
        Binarisation(cameraFrame, 'b', threshValue);

        Mat tmp;
        background.getMat(ACCESS_READ).convertTo(tmp, CV_8U, 0.5, 0);
        visu.convertTo(visu, CV_8U, 0.5, 128);
        subtract(visu, tmp, visu);

        ConcentrationMap(visu, cameraFrame);

        cameraFrame = cameraFrame(ROI);
        visu = visu(ROI);

        imshow("coucou", cameraFrame);

        // Position computation
        out = ObjectPosition(cameraFrame, MINAREA, MAXAREA, visu);


        if(im == 0){ // First frame initialization

            if(out.at(0).size() < NUMBER && out.at(0).size() != 0){
                while((out.at(0).size() - NUMBER) > 0){
                    out.at(0).push_back(Point3f(0, 0, 0));
                    out.at(1).push_back(Point3f(0, 0, 0));
                    out.at(2).push_back(Point3f(0, 0, 0));
                    out.at(3).push_back(Point3f(0, 0, 0));
                }
                outPrev = out;
            }

            else if(out.at(0).empty()){ // Error message
                timer->stop();
                QMessageBox errorBox;
                errorBox.setText("No fish detected! Change parameters!");
                binary ->setChecked(1);
                binary ->isChecked();
                emit grabFrame(visu, cameraFrame);
                errorBox.exec();

            }

            else{
                outPrev = out;
            }
        }



        else if(im > 0){ // Matching fish identity with the previous frame
            identity = CostFunc(outPrev.at(spot), out.at(spot), LENGTH, ANGLE, WEIGHT, LO);
            out.at(0) = Reassignment(outPrev.at(0), out.at(0), identity);
            out.at(1) = Reassignment(outPrev.at(1), out.at(1), identity);
            out.at(2) = Reassignment(outPrev.at(2), out.at(2), identity);
            out.at(3) = Reassignment(outPrev.at(3), out.at(3), identity);
            out.at(spot) = Prevision(outPrev.at(spot), out.at(spot));
            outPrev = out;
        }



        // Visualization & saving
       normalize(visu, visu, 0, 255, NORM_MINMAX);
       applyColorMap(visu, visu, COLORMAP_JET);
       for(unsigned int l = 0; l < out.at(0).size(); l++){
            Point3f coord;
            coord = out.at(spot).at(l);
            arrowedLine(visu, Point(coord.x, coord.y), Point(coord.x + 5*arrowSize*cos(coord.z), coord.y - 5*arrowSize*sin(coord.z)), Scalar(colorMap.at(l).x, colorMap.at(l).y, colorMap.at(l).z), arrowSize, 10*arrowSize, 0);

            if((im > 5)){
                polylines(visu, memory.at(l), false, Scalar(colorMap.at(l).x, colorMap.at(l).y, colorMap.at(l).z), arrowSize, 8, 0);
                memory.at(l).push_back(Point((int)coord.x, (int)coord.y));
                if(im>50){
                    memory.at(l).erase(memory.at(l).begin());
                    }
            }



            // Saving
            coord.x += ROI.tl().x;
            coord.y += ROI.tl().y;
            internalSaving.at(im*NUMBER + l) = coord;
            ofstream savefile;
            savefile.open(savePath, ios::out | ios::app );
            if(im == 0 && l == 0){
                savefile << "xHead" << "   " << "yHead" << "   " << "tHead" << "   "  << "xTail" << "   " << "yTail" << "   " << "tTail"   <<  "   " << "xBody" << "   " << "yBody" << "   " << "tBody"   <<  "   " << "curvature" <<  "   " << "imageNumber" << "   " << "concentration" << '\n';
            }

            savefile << out.at(0).at(l).x + ROI.tl().x << "   " << out.at(0).at(l).y + ROI.tl().y << "   " << out.at(0).at(l).z << "   "  << out.at(1).at(l).x + ROI.tl().x << "   " << out.at(1).at(l).y + ROI.tl().y << "   " << out.at(1).at(l).z  <<  "   " << out.at(2).at(l).x + ROI.tl().y << "   " << out.at(2).at(l).y << "   " << out.at(2).at(l).z <<  "   " << out.at(3).at(l).x <<  "   " << metadata <<  "   "  << out.at(3).at(l).y << "\n";

        }


       emit grabFrame(visu, cameraFrame);


       if(im < files.size() - 1){
           a ++;
           im ++;
           progressBar->setValue(im);
        }

       else{
           timer->stop();
           ReplayButton->show();
           PauseButton ->hide();
           fps ->show();
           fpsField ->show();
           fpsSlider->show();
           trackingSpot->hide();
           trackingSpotLabel->hide();
           im = 0;
           QMessageBox msgBox;
           msgBox.setText("The tracking is done!!! \n You can replay the tracking by clicking the replay button.");
           msgBox.exec();
       }
    }

    catch (const Exception &exc){ // Out of image error
        timer->stop();
        PauseButton ->setText("Play");
        pause = false;
        QMessageBox pathError;
        pathError.setText("The ROI does not fit the image size or there is no object in the image. Please try changing the ROI and the minimal area of the object.");
        x2ROIField->setStyleSheet("background-color: red;");
        y2ROIField->setStyleSheet("background-color: red;");
        pathError.exec();
    }

    catch(const std::out_of_range& oor){ // Out of range error
        timer->stop();
        PauseButton ->setText("Play");
        pause = false;
        QMessageBox pathError;
        pathError.setText("Too many objects in the image that indicated in parameters, try to increase the number of objects or to increase the minimal area of an object.");
        binary ->setChecked(1);
        binary ->isChecked();
        emit grabFrame(visu, cameraFrame);
        pathError.exec();
    }


    catch (const exception &exc){ // Unknown error
        timer->stop();
        PauseButton ->setText("Play");
        pause = false;
        QMessageBox pathError;
        pathError.setText(exc.what());
        pathError.exec();
    }

}



/**
    * @Display: display the originla image and/or the binary image in two Qlabels.
    * @param Mat visu: original image.
    * @param UMat cameraFrame: binary image.
*/
void MainWindow::Display(Mat visu, UMat cameraFrame){


    if (!normal->isChecked() && !binary->isChecked()){
        display->clear();
        display2->clear();
    }

    else if (normal->isChecked() && !binary->isChecked()){
        cvtColor(visu,visu,CV_BGR2RGB);
        Size size = visu.size();

        int w = display->width();
        int h = display->height();

        display->setPixmap(QPixmap::fromImage(QImage(visu.data, visu.cols, visu.rows, visu.step, QImage::Format_RGB888)).scaled(w, h, Qt::KeepAspectRatio));
        display2->clear();
    }

    else if (!normal->isChecked() && binary->isChecked()){ //Display just the binary mask

        int w = display->width();
        int h = display->height();

        display->setPixmap(QPixmap::fromImage(QImage(cameraFrame.getMat(cv::ACCESS_READ).data, cameraFrame.cols, cameraFrame.rows, cameraFrame.step, QImage::Format_Grayscale8)).scaled(w, h, Qt::KeepAspectRatio));
        display2->clear();
    }

    else if (normal->isChecked() && binary->isChecked()){
        cvtColor(visu,visu,CV_BGR2RGB);
        Size size = cameraFrame.size();
        int w = display->width();
        int h = display->height();

        display->setPixmap(QPixmap::fromImage(QImage(visu.data, visu.cols, visu.rows, visu.step, QImage::Format_RGB888)).scaled(w, h, Qt::KeepAspectRatio));
        display2->setPixmap(QPixmap::fromImage(QImage(cameraFrame.getMat(cv::ACCESS_READ).data, cameraFrame.cols, cameraFrame.rows, cameraFrame.step, QImage::Format_Grayscale8)).scaled(w, h, Qt::KeepAspectRatio));
    }

}


/**
    * @Replay: replay the tracking.
    * @To do: update with the new display.
*/
void MainWindow::Replay(){

    // Get parameters
    QString path = pathField->text();
    QString num = numField->text();

    int arrowSize = arrowField-> value();
    QString x1ROI = x1ROIField->text();
    QString x2ROI = x2ROIField->text();
    QString y1ROI = y1ROIField->text();
    QString y2ROI = y2ROIField->text();


    pathField->setDisabled(true);
    numField->setDisabled(true);
    maxAreaField->setDisabled(true);
    minAreaField->setDisabled(true);
    lengthField->setDisabled(true);
    angleField->setDisabled(true);
    loField->setDisabled(true);
    wField->setDisabled(true);
    nBackField->setDisabled(true);
    threshField->setDisabled(true);
    saveField->setDisabled(true);
    x1ROIField->setDisabled(true);
    x2ROIField->setDisabled(true);
    y1ROIField->setDisabled(true);
    y2ROIField->setDisabled(true);





    // Convert parameters
    unsigned int NUMBER = num.toInt(); //number of objects to track
    int FPS = fpsField->value();
    FPS = int((1000.)/(float(FPS)));







    if(imr == 0){ // Initialization

        timerReplay->start(FPS);
        a = files.begin();
        string name = *a;
        progressBar ->setRange(0, files.size());
        visu = imread(name);
        vector<vector<Point> > tmp(NUMBER, vector<Point>());
        memory = tmp;


    }


    string name = *a;
    visu = imread(name);
    timerReplay ->setInterval(FPS);



    // Visualization & saving
   for(unsigned int l = 0; l < NUMBER; l++){
        Point3f coord;
        coord = internalSaving.at(imr*NUMBER + l);
        arrowedLine(visu, Point(coord.x, coord.y), Point(coord.x + 5*arrowSize*cos(coord.z), coord.y - 5*arrowSize*sin(coord.z)), Scalar(colorMap.at(l).x, colorMap.at(l).y, colorMap.at(l).z), arrowSize, 10*arrowSize, 0);

        if((imr > 5)){
            polylines(visu, memory.at(l), false, Scalar(colorMap.at(l).x, colorMap.at(l).y, colorMap.at(l).z), arrowSize, 8, 0);
            memory.at(l).push_back(Point((int)coord.x, (int)coord.y));
            if(imr>50){
                memory.at(l).erase(memory.at(l).begin());
                }
        }
    }



   cvtColor(visu,visu,CV_BGR2RGB);

   int w = display->width();
   int h = display->height();

   display->setPixmap(QPixmap::fromImage(QImage(visu.data, visu.cols, visu.rows, visu.step, QImage::Format_RGB888)).scaled(w, h, Qt::KeepAspectRatio));
   display2->clear();

    if(imr < files.size() - 1){
        a ++;
        imr ++;
        progressBar->setValue(imr);
     }

    else{
        timerReplay->stop();
        imr = 0;
    }

}


/**
    * @Write: write a text file with the parameters when clicking on "set as default" button.
*/
void MainWindow::Write(){

    QString filename = "conf.txt";
    QFile file(filename);
    if(file.open(QIODevice::ReadWrite))
    {
        QTextStream stream( &file );
        stream << pathField->text()<< endl;
        stream << numField->text()<< endl;
        stream << maxAreaField->text()<< endl;
        stream << minAreaField->text()<< endl;
        stream << lengthField->text()<< endl;
        stream << angleField->text()<< endl;
        stream << loField->text()<< endl;
        stream << wField->text()<< endl;
        stream << nBackField->text()<< endl;
        stream << threshField->text()<< endl;
        stream << saveField->text()<< endl;
        stream << x1ROIField->text()<< endl;
        stream << x2ROIField->text()<< endl;
        stream << y1ROIField->text()<< endl;
        stream << y2ROIField->text()<< endl;
    }
}



/**
    * @Reset: reset the program.
*/
void MainWindow::Reset()
{
    qDebug() << "Performing application reboot...";
    qApp->exit(MainWindow::EXIT_CODE_REBOOT );
}


void MainWindow::checkPath(QString path){
    string folder = path.toStdString();
    vector<String> files;
    try{ // Found images in the given path
        glob(folder, files, false);
        Mat img = imread(files.at(0), IMREAD_GRAYSCALE);
        pathField->setStyleSheet("background-color: green;");

        if(x2ROIField->text().toInt() == 0){ // Default set of the ROI width
            x2ROIField ->setText(QString::number(img.cols));
        }

        if(y2ROIField->text().toInt() == 0){ // Defaut set of the ROI height
            y2ROIField ->setText(QString::number(img.rows));
        }

    }
    catch(...){ // No image in the given path
        pathField->setStyleSheet("background-color: red;");
    }


}


/**
    * @setSavePath: auto-writting the output txt file in path_of_images/tracking.txt
*/
void MainWindow::setSavePath(QString path){
    string folder = path.toStdString();
    size_t found = folder.find("/*");
    string savePath = folder.substr (0,found) + "/tracking.txt";
    saveField->setText(QString::fromStdString(savePath));
}


MainWindow::~MainWindow()
{
    delete ui;
}

