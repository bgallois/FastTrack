#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QProgressBar>
#include <QImage>
#include <QMessageBox>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <QCheckBox>
#include <QSlider>
#include <QLCDNumber>
#include <QComboBox>
#include <QThread>
#include <QSpinBox>
#include <QLayout>
#include <QWidget>



using namespace std;
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static int const EXIT_CODE_REBOOT = -123456789;



private:

    Ui::MainWindow *ui;
    QThread *thread;
    QPushButton *QuitButton;
    QPushButton *GoButton;
    QPushButton *PauseButton;
    QPushButton *DefaultButton;
    QPushButton *ResetButton;
    QPushButton *ReplayButton;
    QCheckBox *normal;
    QCheckBox *binary;
    QMessageBox *welcomeBox;
    QString version = "v3.6.1";

    QWidget *central;
    QLayout *layout;
    QLabel *refreshRateLabel;
    QLabel *path;
    QLabel *numLabel;
    QLabel *maxArea;
    QLabel *minArea;
    QLabel *length;
    QLabel *lo;
    QLabel *w;
    QLabel *angle;
    QLabel *nBack;
    QLabel *thresh;
    QLabel *save;
    QLabel *x1ROI;
    QLabel *y1ROI;
    QLabel *x2ROI;
    QLabel *y2ROI;
    QLabel *arrow;
    QLabel *fps;
    QLabel *trackingSpotLabel;

    QLineEdit *pathField;
    QLineEdit *numField;
    QLineEdit *maxAreaField;
    QLineEdit *minAreaField;
    QLineEdit *lengthField;
    QLineEdit *loField;
    QLineEdit *wField;
    QLineEdit *angleField;
    QLineEdit *nBackField;
    QLineEdit *threshField;
    QLineEdit *saveField;
    QLineEdit *x1ROIField;
    QLineEdit *y1ROIField;
    QLineEdit *x2ROIField;
    QLineEdit *y2ROIField;
    QLCDNumber *arrowField;
    QLCDNumber *fpsField;

    QLabel *display;
    QLabel *display2;
    QTimer *timer;
    QTimer *timerDisplay;
    QTimer *timerReplay;
    QProgressBar *progressBar;
    QSlider *arrowSlider;
    QSlider *fpsSlider;
    QComboBox *trackingSpot;

    vector<String> files;
    vector <String>::iterator a;

    UMat cameraFrame;
    Mat visu;
    UMat img0;
    UMat background;
    Mat concentrationImg;
    vector<vector<Point> > memory;
    vector<Point3f> colorMap;
    vector<vector<Point3f> > out;
    vector<vector<Point3f> > outPrev;
    vector<int> identity;
    unsigned int im;
    unsigned int imr;
    ofstream savefile;
    vector<Point3f> internalSaving;
    int spot;
    int arrowSize;
    int MAXAREA;
    int MINAREA;
    int x1;
    int y1;
    int x2;
    int y2;
    double LENGTH;
    double ANGLE;
    unsigned int NUMBER;
    double WEIGHT;
    double LO;
    int nBackground;
    int threshValue;
    string savePath;
    bool pause;

public slots:
    void Go();
    void Write();
    void Reset();
    void Display(Mat visu, UMat cameraFrame);
    void Replay();
    void UpdateParameters();
    void PlayPause();
    void checkPath(QString);

signals:
    void grabFrame(Mat visu, UMat cameraFrame);
};

#endif // MAINWINDOW_H
