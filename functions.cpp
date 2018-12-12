/**############################################################################################
								    functions.cpp
								    Purpose: Function use in the main.cpp

								    @author Benjamin GALLOIS
										@email benjamin.gallois@upmc.fr
										@website benjamin-gallois.fr
								    @version 2.0
										@date 2018
###############################################################################################*/

/*
     .-""L_        		     .-""L_
;`, /   ( o\ 			;`, /   ( o\
\  ;    `, /   			\  ;    `, /
;_/"`.__.-"				;_/"`.__.-"

     .-""L_        		     .-""L_
;`, /   ( o\ 			;`, /   ( o\
\  ;    `, /   			\  ;    `, /
;_/"`.__.-"				;_/"`.__.-"


*/



#include "functions.h"
#include "Hungarian.h"

using namespace cv;
using namespace std;




/**
  * @CurvatureCenter Computes the center of the curvature defined as the intersection of the minor axis of the head ellipse with the minor axis of the tail ellipse.
  * @param Point3f tail: parameter of the tail x, y and angle of orientation
	* @param Point3f tail: parameter of the head x, y and angle of orientation
  * @return Point2f: coordinates of the center of the curvature
*/
Point2f Tracking::curvatureCenter(Point3f tail, Point3f head){

	Point2f center;

	Point p1 = Point(tail.x + 10*cos(tail.z + 0.5*M_PI), tail.y + 10*sin(tail.z + 0.5*M_PI));
	Point p2 = Point(head.x + 10*cos(head.z + 0.5*M_PI), head.y + 10*sin(head.z + 0.5*M_PI));

	double a = (tail.y - p1.y)/(tail.x - p1.x);
	double c = (head.y - p2.y)/(head.x - p2.x);
	double b = tail.y - a*tail.x;
	double d = head.y - c*head.x;

	if(a*b == 0){ // Determinant == 0, no unique solution
		center = Point(0, 0); // TO CHECK
	}

	else{ // Unique solution
		center = Point((b + d)/(c - a), a*((b +d)/(c - a)) + b);
	}

   return center;
}




/**
  * @Curvature Computes the radius of curvature of the fish defined as the inverse of the mean distance between each pixel of the fish and the center of the curvature.
  * @param Point2f center: center of the curvature
	* @param mat image: binary image CV_8U
  * @return double: radius of curvature
*/
double Tracking::curvature(Point2f center , Mat image){

	double d = 0;
	double count = 0;
  
  image.forEach<uchar>
(
  [&d, &count, center](uchar &pixel, const int position[]) -> void
  {
			if(pixel == 255){ // if inside object
        d += pow(pow(center.x - float(position[0]), 2) + pow(center.y - float(position[1]), 2), 0.5);
          count += 1;
    }
  }
); 
	return count/d;
}



/**
  * @Modul usual modulo 2*PI of an angle.
  * @param double angle: input angle
  * @return double: output angle
*/
double Modul(double angle)
{
    return angle - 2*M_PI * floor(angle/(2*M_PI));
}




/**
  * @objectInformation computes the equivalente ellipse of an object and it direction
  * @param Mat image: binary image CV_8U
  * @return vector<double>: [x, y, orientation]
  * @note Computes the orientation but not the direction.
*/
vector<double> Tracking::objectInformation(UMat image) {


    Moments moment = moments(image);

    double x = moment.m10/moment.m00;
    double y = moment.m01/moment.m00;

    double i = moment.mu20;
    double j = moment.mu11;
    double k = moment.mu02;


    double orientation = 0.5 * atan((2*j)/(i-k)) + (i<k)*(M_PI*0.5);
  	orientation += 2*M_PI*(orientation<0);
	  orientation = 2*M_PI - orientation;
	
    vector<double> param {x, y, orientation};

  	return param;
}




/**
  * @objectDirection computes the direction of the object from the object and the orientation.
  * @param Mat image: binary image CV_8U
  * @information vector<double>: &[x, y, orientation]
  * @center Point: barycenter of the object.
  * @return true if the direction is orientation + pi.
*/
bool Tracking::objectDirection(UMat image, Point center, vector<double> &information) {
    
    vector<double> tmpMat;
    reduce(image, tmpMat, 0, REDUCE_SUM);
    
    double skew = accumulate(tmpMat.begin(), tmpMat.begin() + int(center.x), 0) - accumulate(tmpMat.begin() + int(center.x), tmpMat.end(), 0);
		
    if(skew > 0){
			information.at(2) -= M_PI;
			information.at(2) = Modul(information.at(2));
      return true;
		}
    return false;
	}




/**
  * @BackgroundExtraction extracts the background of a film with moving object by projection
  * @param vector<String> files: array with the path of images
  @param double n: number of frames to average
  * @return UMat: background image of a movie
*/
UMat Tracking::backgroundExtraction(const vector<String> &files, double n){

    UMat background;
    UMat img0;
    imread(files[0], IMREAD_GRAYSCALE).copyTo(background);
    background.setTo(0);
    imread(files[0], IMREAD_GRAYSCALE).copyTo(img0);
    background.convertTo(background, CV_32FC1);
    img0.convertTo(img0, CV_32FC1);
    int step = files.size()/n;
    UMat cameraFrameReg;
    Mat H;
    int count = 0;

	for(unsigned int i = 0; i < files.size(); i += step){
        imread(files[i], IMREAD_GRAYSCALE).copyTo(cameraFrameReg);
        cameraFrameReg.convertTo(cameraFrameReg, CV_32FC1);
        Point2d shift = phaseCorrelate(cameraFrameReg, img0);
        H = (Mat_<float>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
				warpAffine(cameraFrameReg, cameraFrameReg, H, cameraFrameReg.size());
				accumulate(cameraFrameReg, background);
        count ++;
	}
  background.convertTo(background, CV_8U, 1./count);

	return background;
}




/**
  * @Registration makes the registration of a movie by phase correlation
  * @param UMat imageReference: reference image for the registration, one channel
	* @param UMat frame: image to register
*/
void Tracking::registration(UMat imageReference, UMat &frame){

	frame.convertTo(frame, CV_32FC1);
	imageReference.convertTo(imageReference, CV_32FC1);
	Point2d shift = phaseCorrelate(frame, imageReference);
  Mat H = (Mat_<float>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
	warpAffine(frame, frame, H, frame.size());
  frame.convertTo(frame, CV_8U);
}




/**
  * @Binarisation binarizes the image by an Otsu threshold
  * @param UMat frame: image to binarized
	* @param char backgroundColor: 'b' if the background is black, 'w' is the background is white
*/
void Tracking::binarisation(UMat& frame, char backgroundColor, int value){

  frame.convertTo(frame, CV_8U);

	if(backgroundColor == 'b'){
        threshold(frame, frame, value, 255, THRESH_BINARY);
	}

	if(backgroundColor == 'w'){
        threshold(frame, frame, value, 255, THRESH_BINARY_INV);
	}
}




/**
  * @ObjectPosition Computes positions of multiples objects of size between min and max size by finding contours
  * @param UMat frame: binary image CV_8U
	* @param int minSize: minimal size of the object
	* @param int maxSize: maximal size of the object
  * @return vector<vector<Point3f>>: {head parameters, tail parameters, global parameter}, {head/tail parameters} = {x, y, orientation}, {global parameter} = {curvature, 0, 0}
*/
vector<vector<Point3f>> Tracking::objectPosition(UMat frame, int minSize, int maxSize){

	vector<vector<Point> > contours;
	vector<Point3f> positionHead;
	vector<Point3f> positionTail;
	vector<Point3f> positionFull;
	vector<Point3f> globalParam;
	UMat dst;
	Rect roiFull, bbox;
	UMat RoiFull, RoiHead, RoiTail, rotate;
	Mat rotMatrix, p, pp;
	vector<double> parameter;
	vector<double> parameterHead;
	vector<double> parameterTail;
	Point2f radiusCurv;

	findContours(frame, contours, RETR_LIST, CHAIN_APPROX_NONE);

	for (unsigned int i = 0; i < contours.size(); i++){

			if(contourArea(contours[i]) > minSize && contourArea(contours[i]) < maxSize){ // Only select objects minArea << objectArea <<maxArea

            // Draw the object in a temporary black image avoiding to select a 
            // part of another object if two objects are very close.
						dst = UMat::zeros(frame.size(), CV_8U);
						drawContours(dst, contours, i, Scalar(255, 255, 255), FILLED,8); 
					
          	
            // Computes the x, y and orientation of the object, in the
            // frame of reference of ROIFull image.
            roiFull = boundingRect(contours[i]);
						RoiFull = dst(roiFull);
						parameter = objectInformation(RoiFull);
            

            // Rotates the image without croping and computes the direction of the object.
            Point center = Point(0.5*RoiFull.cols, 0.5*RoiFull.rows);
            rotMatrix = getRotationMatrix2D(center, -(parameter.at(2)*180)/M_PI, 1);
            bbox = RotatedRect(center, RoiFull.size(), -(parameter.at(2)*180)/M_PI).boundingRect();
            rotMatrix.at<double>(0,2) += bbox.width*0.5 - center.x;
            rotMatrix.at<double>(1,2) += bbox.height*0.5 - center.y;
            warpAffine(RoiFull, rotate, rotMatrix, bbox.size());


           // Computes the coordinate of the center of mass of the fish in the rotated
           // image frame of reference.
            p = (Mat_<double>(3,1) << parameter.at(0), parameter.at(1), 1);
            pp = rotMatrix * p;

					
            // Computes the direction of the object. If objectDirection return true, the
            // head is at the left and the tail at the right.
            Rect roiHead, roiTail;   	
            if ( objectDirection(rotate, Point(pp.at<double>(0,0), pp.at<double>(1,0)), parameter) ) {

              // Head ellipse. Parameters in the frame of reference of the RoiHead image.
              roiHead = Rect(0, 0, pp.at<double>(0,0), rotate.rows);
              RoiHead = rotate(roiHead);
              parameterHead = objectInformation(RoiHead);

              // Tail ellipse. Parameters in the frame of reference of ROITail image.
              roiTail = Rect(pp.at<double>(0,0), 0, rotate.cols-pp.at<double>(0,0), rotate.rows);
              RoiTail = rotate(roiTail);
              parameterTail = objectInformation(RoiTail);

            }
            else {
              // Head ellipse. Parameters in the frame of reference of the RoiHead image.
              roiHead = Rect(pp.at<double>(0,0), 0, rotate.cols-pp.at<double>(0,0), rotate.rows);
              RoiHead = rotate(roiHead);
              parameterHead = objectInformation(RoiHead);

              // Tail ellipse. Parameters in the frame of reference of RoiTail image.
              roiTail = Rect(0, 0, pp.at<double>(0,0), rotate.rows);
              RoiTail = rotate(roiTail);
              parameterTail = objectInformation(RoiTail);
            }


            // Gets all the parameter in the frame of reference of RoiFull image.
						invertAffineTransform(rotMatrix, rotMatrix);
						p = (Mat_<double>(3,1) << parameterHead.at(0) + roiHead.tl().x,parameterHead.at(1) + roiHead.tl().y, 1);
						pp = rotMatrix * p;

						double xHead = pp.at<double>(0,0) + roiFull.tl().x;
						double yHead = pp.at<double>(1,0) + roiFull.tl().y;
						double angleHead = parameterHead.at(2) - M_PI*(parameterHead.at(2) > M_PI);
						angleHead = Modul(angleHead + parameter.at(2) + M_PI*(abs(angleHead) > 0.5*M_PI)); // Computes the direction

						p = (Mat_<double>(3,1) << parameterTail.at(0) + roiTail.tl().x, parameterTail.at(1) + roiTail.tl().y, 1);
						pp = rotMatrix * p;
						double xTail = pp.at<double>(0,0) + roiFull.tl().x;
						double yTail = pp.at<double>(1,0) + roiFull.tl().y;
						double angleTail = parameterTail.at(2) - M_PI*(parameterTail.at(2) > M_PI);
						angleTail = Modul(angleTail + parameter.at(2) + M_PI*(abs(angleTail) > 0.5*M_PI)); // Computes the direction

						// Computes the curvature of the object as the invert of all distances from each
            // pixels of the fish and the intersection of the minor axis off tail and head ellipse.
						double curv = 1./1e-16;
						radiusCurv = curvatureCenter(Point3f(xTail, yTail, angleTail), Point3f(xHead, yHead, angleHead));
						if(radiusCurv.x != NAN){ //
						            curv = curvature(radiusCurv, RoiFull.getMat(ACCESS_READ));
						}


						positionHead.push_back(Point3f(xHead, yHead, angleHead));
						positionTail.push_back(Point3f(xTail, yTail, angleTail));
						positionFull.push_back(Point3f(parameter.at(0) + roiFull.tl().x, parameter.at(1) + roiFull.tl().y, parameter.at(2)));

						globalParam.push_back(Point3f(curv, 0, 0));
						}
   }

	vector<vector<Point3f>> out = {positionHead, positionTail, positionFull, globalParam};
	return out;
}





/**
  * @CostFunc computes the cost function and use a global optimization association to associate target between frame. Method adapted from: "An
							effective and robust method for Tracking multiple fish in video image based on fish head detection" YQ Chen et al.
							Use the Hungarian method implemented by Cong Ma, 2016 "https://github.com/mcximing/hungarian-algorithm-cpp" adapted from the matlab
							implementation by Markus Buehren "https://fr.mathworks.com/matlabcentral/fileexchange/6543-functions-for-the-rectangular-assignment-problem".
  * @param vector<Point3f> prevPos: sorted vector of object parameters,vector of points (x, y, orientation).
	* @param vector<Point3f> pos: non-sorted vector of object parameters,vector of points (x, y, orientation) that we want to sort accordingly to prevPos to identify each object.
	* @param double length: maximal displacement of an object between two frames.
	* @param double angle: maximal difference angle of an object direction between two frames.
	* @return vector<int>: the assignment vector of the new index position.
*/
vector<int> Tracking::costFunc(vector<Point3f> prevPos, vector<Point3f> pos, const double LENGTH, const double ANGLE, const double WEIGHT, const double LO){


	int n = prevPos.size();
	int m = pos.size();
	double c = -1;
	vector<vector<double>> costMatrix(n, vector<double>(m));

	for(int i = 0; i < n; ++i){

		Point3f prevCoord = prevPos.at(i);
		for(int j = 0; j < m; ++j){
			Point3f coord = pos.at(j);
            double d = pow(pow(prevCoord.x - coord.x, 2) + pow(prevCoord.y - coord.y, 2), 0.5);
            if(d < LO){
                c = WEIGHT*(d/LENGTH) + (1 - WEIGHT)*((abs(Modul(prevCoord.z - coord.z + M_PI) - M_PI))/(ANGLE)); //cost function
				costMatrix[i][j] = c;
			}
            else if (d > LO){
				costMatrix[i][j] = 2e53;
			}

		}
	}

	// Hungarian algorithm to solve the assignment problem O(n**3)
	HungarianAlgorithm HungAlgo;
	vector<int> assignment;
	HungAlgo.Solve(costMatrix, assignment);

	return assignment;
}




/**
  * @Reassignment Resamples a vector accordingly to a new index.
  * @param vector<Point3f> output: output vector of size n
  * @param vector<Point3f> input: input vector of size m <= n
  * @param vector<int> assignment: vector with the new index that will be used to resample the input vector
  * @return vector<Point3f>: output vector of size n.
*/
vector<Point3f> Tracking::reassignment(vector<Point3f> output, vector<Point3f> input, vector<int> assignment){

	vector<Point3f> tmp = output;
	unsigned int n = output.size();
	unsigned int m = input.size();


	if(m == n){ // Same number of targets in two consecutive frames
		for(unsigned int i = 0; i < n; i++){
			tmp.at(i) = input.at(assignment.at(i));
		}
	}

	else if(m > n){// More target in current frame than in previous one
		for(unsigned int i = 0; i < n; i++){
			tmp.at(i) = input.at(assignment.at(i));
		}
	}

	else if(m < n){// Fewer target in current frame than in the previous one
		for(unsigned int i = 0; i < n; i++){
			if(assignment.at(i) != -1){
				tmp.at(i) = input.at(assignment.at(i));
			}
		}
	}

	else{
		cout << "association error" << '\n';
	}

	input = tmp;

	return input;
}




/**
  * @Reassignment Resamples a vector accordingly to a new index.
  * @param vector<Point3f> output: output vector of size n
  * @param vector<Point3f> input: input vector of size m <= n
  * @param vector<int> assignment: vector with the new index that will be used to resample the input vector
  * @return vector<Point3f>: output vector of size n.
*/
vector<Point3f> Tracking::prevision(vector<Point3f> past, vector<Point3f> present){

	double l = 0;
	for(unsigned int i = 0; i < past.size(); i++){
		if(past.at(i) != present.at(i)){
			l = pow(pow(past.at(i).x - present.at(i).x, 2) + pow(past.at(i).y - present.at(i).y, 2), 0.5);
			break;
		}
	}



	for(unsigned int i = 0; i < past.size(); i++){
		if(past.at(i) == present.at(i)){
			present.at(i).x += l*cos(present.at(i).z);
			present.at(i).y -= l*sin(present.at(i).z);
		}
	}
	return present;
}





/**
  * @Color computes a color map.
  * @param int number: number of colors

  * @return vector<Point3f>: RGB color
*/
vector<Point3f> Tracking::color(int number){

	double a, b, c;
	vector<Point3f> colorMap;
	srand (time(NULL));
	for (int j = 0; j<number; ++j){
		a = rand() % 255;
		b = rand() % 255;
		c = rand() % 255;

		colorMap.push_back(Point3f(a, b, c));
	}

	return colorMap;
}





void Tracking::imageProcessing(){

    imread(m_files.at(m_im), IMREAD_GRAYSCALE).copyTo(m_visuFrame);
    if(statusRegistration){
      registration(m_background, m_visuFrame);
    }

    (statusBinarisation) ? (subtract(m_background, m_visuFrame, m_binaryFrame)) : (subtract(m_visuFrame, m_background, m_binaryFrame));

    binarisation(m_binaryFrame, 'b', param_thresh);

    if (param_dilatation != 0) {
      Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*param_dilatation + 1, 2*param_dilatation + 1 ), Point( param_dilatation, param_dilatation ) );
      dilate(m_binaryFrame, m_binaryFrame, element);
    }

    if(m_ROI.width != 0 ) {
      m_binaryFrame = m_binaryFrame(m_ROI);
      m_visuFrame = m_visuFrame(m_ROI);
    }

    m_out = objectPosition(m_binaryFrame, param_minArea, param_maxArea);

    vector<int> identity = costFunc(m_outPrev.at(param_spot), m_out.at(param_spot), param_len, param_angle, param_weight, param_lo);
    for (unsigned int i = 0; i < m_out.size(); i++) {
      m_out.at(i) = reassignment(m_outPrev.at(i), m_out.at(i), identity);
    }
    cvtColor(m_visuFrame, m_visuFrame, COLOR_GRAY2RGB);
    // Visualisation

    for(unsigned int l = 0; l < m_out.at(0).size(); l++){
      Point3f coord = m_out.at(param_spot).at(l);
      arrowedLine(m_visuFrame, Point(coord.x, coord.y), Point(coord.x + 5*param_arrowSize*cos(coord.z), coord.y - 5*param_arrowSize*sin(coord.z)), Scalar(m_colorMap.at(l).x, m_colorMap.at(l).y, m_colorMap.at(l).z), param_arrowSize, 10*param_arrowSize, 0);

      if((m_im > 5)){ // Faudra refaire un buffer correct
        polylines(m_visuFrame, m_memory.at(l), false, Scalar(m_colorMap.at(l).x, m_colorMap.at(l).y, m_colorMap.at(l).z), param_arrowSize, 8, 0);
        m_memory.at(l).push_back(Point((int)coord.x, (int)coord.y));
      if(m_im > 50){
          m_memory.at(l).erase(m_memory.at(l).begin());
        }
      }

      // Saving
      coord.x += m_ROI.tl().x;
      coord.y += m_ROI.tl().y;

      m_savefile << m_out.at(0).at(l).x + m_ROI.tl().x << '	' << m_out.at(0).at(l).y + m_ROI.tl().y << '	' << m_out.at(0).at(l).z << '	'  << m_out.at(1).at(l).x + m_ROI.tl().x << '	' << m_out.at(1).at(l).y + m_ROI.tl().y << '	' << m_out.at(1).at(l).z  <<  '	' << m_out.at(2).at(l).x + m_ROI.tl().x << '	' << m_out.at(2).at(l).y  + m_ROI.tl().y << '	' << m_out.at(2).at(l).z <<  '	' << m_out.at(3).at(l).x <<  '	' << m_im << "\n";

    }
  
    m_im ++;
    m_outPrev = m_out;
    
    // Sending rate of images
    if ( (timer->elapsed() - m_displayTime) > 40) {
      emit(newImageToDisplay(m_visuFrame, m_binaryFrame));
      m_displayTime = timer->elapsed();
    }
    if(m_im + 1 > int(m_files.size())){
      m_savefile.flush();
      m_outputFile.close();
      emit(finished());
    qInfo() << timer->elapsed() << endl;
    }
    else {
    QTimer::singleShot(0, this, SLOT(imageProcessing()));
    }
}




Tracking::Tracking(string path) {
  m_path = path;
}


void Tracking::startProcess() {
  
  // Find image format
  QList<QString> extensions = { "pgm", "png", "jpeg", "jpg", "tiff", "tif", "bmp", "dib", "jpe", "jp2", "webp", "pbm", "ppm", "sr", "ras", "tif" };
  QDirIterator it(QString::fromStdString(m_path), QStringList(), QDir::NoFilter);
  QString extension;
  while (it.hasNext()) {
    extension = it.next().section('.', -1);
    if( extensions.contains(extension) ) break;
  }
  
  try{
    m_path += + "*." + extension.toStdString();
    glob(m_path, m_files, false); // Get all path to frames
    statusPath = true;
    m_im = 0;
  }
  catch(...){
    statusPath = false;
    emit(finished());
  }
  sort(m_files.begin(), m_files.end());

  m_background = backgroundExtraction(m_files, param_nBackground);
  m_colorMap = color(param_n);
  vector<vector<Point> > tmp(param_n, vector<Point>());
  m_memory = tmp;


  // First frame
  imread(m_files.at(0), IMREAD_GRAYSCALE).copyTo(m_visuFrame);
  
  (statusBinarisation) ? (subtract(m_background, m_visuFrame, m_binaryFrame)) : (subtract(m_visuFrame, m_background, m_binaryFrame));

  binarisation(m_binaryFrame, 'b', param_thresh);

  if (param_dilatation != 0) {
    Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*param_dilatation + 1, 2*param_dilatation + 1 ), Point( param_dilatation, param_dilatation ) );
    dilate(m_binaryFrame, m_binaryFrame, element);
  }

  if (m_ROI.width != 0){
    m_binaryFrame = m_binaryFrame(m_ROI);
    m_visuFrame = m_visuFrame(m_ROI);
  }

  m_out= objectPosition(m_binaryFrame, param_minArea, param_maxArea);
  
  // if less objects detected than indicated
  while( (int(m_out.at(0).size()) - param_n) < 0 ){
      for(unsigned int i = 0; i < m_out.size(); i++){
        m_out.at(i).push_back(Point3f(0,0,0));
      }
  }

  // if more objects detected than indicated
  while( (m_out.at(0).size() - param_n) > 0 ){
      for(unsigned int i = 0; i < m_out.size(); i++){
        m_out.at(i).pop_back();
      }
  }
    
  cvtColor(m_visuFrame, m_visuFrame, COLOR_GRAY2RGB);
  
  // Initialize output file and stream
  m_outputFile.setFileName(QString::fromStdString(m_path).section("*",0,0) + "Tracking_Result" + QDir::separator() + "tracking.txt" );
  if(!m_outputFile.open(QFile::WriteOnly | QFile::Text)){
    qInfo() << "Error opening folder";
  }
  m_savefile.setDevice(&m_outputFile);

  // Saving
  m_savefile << "xHead" << '\t' << "yHead" << '\t' << "tHead" << '\t'  << "xTail" << '\t' << "yTail" << '\t' << "tTail"   << '\t'  << "xBody" << '\t' << "yBody" << '\t' << "tBody"   << '\t'  << "curvature" << '\t'  << "imageNumber" << "\n";
  for(unsigned int l = 0; l < m_out.at(0).size(); l++){
    Point3f coord = m_out.at(param_spot).at(l);      
    arrowedLine(m_visuFrame, Point(coord.x, coord.y), Point(coord.x + 5*param_arrowSize*cos(coord.z), coord.y - 5*param_arrowSize*sin(coord.z)), Scalar(m_colorMap.at(l).x, m_colorMap.at(l).y, m_colorMap.at(l).z), param_arrowSize, 10*param_arrowSize, 0);

    if((m_im > 5)){ // Faudra refaire un buffer correct
      polylines(m_visuFrame, m_memory.at(l), false, Scalar(m_colorMap.at(l).x, m_colorMap.at(l).y, m_colorMap.at(l).z), param_arrowSize, 8, 0);
      m_memory.at(l).push_back(Point((int)coord.x, (int)coord.y));
    if(m_im > 50){
        m_memory.at(l).erase(m_memory.at(l).begin());
      }
    }

    // Saving
    coord.x += m_ROI.tl().x;
    coord.y += m_ROI.tl().y;

    m_savefile << m_out.at(0).at(l).x + m_ROI.tl().x << '	' << m_out.at(0).at(l).y + m_ROI.tl().y << '	' << m_out.at(0).at(l).z << '	'  << m_out.at(1).at(l).x + m_ROI.tl().x << '	' << m_out.at(1).at(l).y + m_ROI.tl().y << '	' << m_out.at(1).at(l).z  <<  '	' << m_out.at(2).at(l).x + m_ROI.tl().x << '	' << m_out.at(2).at(l).y  + m_ROI.tl().y << '	' << m_out.at(2).at(l).z <<  '	' << m_out.at(3).at(l).x <<  '	' << m_im << "\n";

  }
  m_outPrev = m_out;
  m_im ++;
  connect(this, SIGNAL(finishedProcessFrame()), this, SLOT(imageProcessing()));
  timer = new QElapsedTimer();


  timer->start();
  m_displayTime = 0;
  emit(newImageToDisplay(m_visuFrame, m_binaryFrame));
  emit(finishedProcessFrame());
}

void Tracking::updatingParameters(const QMap<QString, QString> &parameterList) {

  param_n = parameterList.value("Object number").toInt();
  param_maxArea = parameterList.value("Maximal size").toInt();
  param_minArea = parameterList.value("Minimal size").toInt();
  param_spot = parameterList.value("Spot to track").toInt();
  param_len = parameterList.value("Maximal length").toDouble();
  param_angle = parameterList.value("Maximal angle").toDouble();
  param_weight = parameterList.value("Weight").toDouble();
  param_lo = parameterList.value("Maximum occlusion").toDouble();
  param_arrowSize = parameterList.value("Arrow size").toInt();

  param_thresh = parameterList.value("Binary threshold").toInt();
  param_nBackground = parameterList.value("Number of images background").toDouble();
  param_x1 = parameterList.value("ROI top x").toInt();
  param_y1 = parameterList.value("ROI top y").toInt();
  param_x2 = parameterList.value("ROI bottom x").toInt();
  param_y2 = parameterList.value("ROI bottom y").toInt();
  m_ROI = Rect(param_x1, param_y1, param_x2 - param_x1, param_y2 - param_y1);
  statusRegistration = (parameterList.value("Registration") == "yes") ? true : false;
  statusBinarisation = (parameterList.value("Light background") == "yes") ? true : false;
  param_dilatation = parameterList.value("Dilatation").toInt();

  qInfo() << "Parameters updated" << endl;
}


Tracking::~Tracking() {
}
