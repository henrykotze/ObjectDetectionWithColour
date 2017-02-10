
#include <iostream>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <stdlib.h>
#include <stdio.h>
#include <opencv/cxcore.h>
#include <stdbool.h>
#include "opencv/ml.h"
#include <time.h>
#include <math.h> //maybe replace with c++ equivalent library
#include "pca9685.h"
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
// #include <netinet>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#define PIN_BASE 300
#define MAX_PWM 4096
#define HERTZ 50
#define DEBUG 0


class Camera{
public:
  Camera(double x = 1.5, double y = 1.5, int pinx = 14, int piny = 15):x_axis(x), y_axis(y), pin_x(pinx), pin_y(piny){} //these values is used with Sunfounder Smart Kit guide
  Camera(const Camera& copy): y_axis(copy.y_axis), x_axis(copy.x_axis), pin_x(copy.pin_x), pin_y(copy.pin_y){}
  double getX_pos(){return x_axis;}
  double getY_pos(){return y_axis;}


  void move_x_axis(double xDistance);
  void move_y_axis(double yDistance);
private:
  double x_axis;
  double y_axis; //preset these values to their origin

  int pin_x;    //pin for x-axis movement for camera
  int pin_y;    //pin for y-axis movement for camera

};
struct CompareHolder {
	char name;
	CvPoint MaxLoc;
	double MaxAcc;
	CvSeq* Contour;
};
//
// struct Cam{
//   int pin_x;
//   int pin_y;
//   double x_axis;
//   double y_axis;
//
// };

//PRESET ALL THESE VALUES TO SEE A FROM START A SPECIFIC COLOUR
int R_lower = 50;
int G_lower = 10;
int B_lower = 60;

int R_higher = 150;
int G_higher = 255;
int B_higher = 255;

int Threshold = 0;
int max_value = 0;

int ErodeIterations  = 1;
int DilateIterations = 1;

bool DrawRect = false;

CvScalar OuterColour;
CvScalar InnerColour;

CvScalar LowerBound;
CvScalar UpperBound;

CvCapture* Input = NULL;

IplImage* frame = NULL;
IplImage* RGB_alter = NULL;
IplImage* Gray = NULL;
IplImage* DetectedObject = NULL;

void SetLowerRValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	R_lower = pos;
	LowerBound = cvScalar(R_lower, G_lower, B_lower, 0);
}
void SetLowerGValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	G_lower = pos;
	LowerBound = cvScalar(R_lower, G_lower, B_lower, 0);
}
void SetLowerBValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	B_lower = pos;
	LowerBound = cvScalar(R_lower, G_lower, B_lower, 0);
}
void SetHigherRValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	R_higher = pos;
	UpperBound = cvScalar(R_higher, G_higher, B_higher, 0);
}
void SetHigherGValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	G_higher = pos;
	UpperBound = cvScalar(R_higher, G_higher, B_higher, 0);
}
void SetHigherBValue(int pos)
{//	cvSetCaptureProperty(Input, CV_CAP_PROP_POS_FRAMES, pos);
	B_higher = pos;
	UpperBound = cvScalar(R_higher, G_higher, B_higher, 0);
}
void SetThreshold(int pos){
	Threshold = pos;
}
void SetThresholdMaxValue(int pos)
{
	max_value = pos;
}
void SetNumOfErodeIteration(int pos)
{
	ErodeIterations = pos;
}
void SetNumOfDilateIteration(int pos)
{
	DilateIterations = pos;
}
void CalcAngleAndDist( CvPoint origin, CvPoint midPnt, CvPoint prevMidPnt, double x_axisUnit, double y_axisUnit, Camera& cam)
{

    double angle;

		if(midPnt.x - origin.x != 0){

		double Distance =  pow( pow( (origin.x - midPnt.x), 2) + pow( (origin.y - midPnt.y), 2), 0.5);
		double deltaY = -1*(midPnt.y - origin.y);
		double deltaX = midPnt.x - origin.x;

    if(DEBUG){
      if(deltaY > 0 && deltaX > 0){ //First quadrant
        printf("1 \n" );
        angle = atan2(deltaY, deltaX);
      }
      else if(deltaY > 0 && deltaX < 0){  //Second quadrant
        printf("2 \n");
        angle = atan2(deltaY, deltaX);
      }
      else if(deltaY < 0 && deltaX < 0){  //Third Quadrant
        angle = 2*M_PI + atan2(deltaY, deltaX);
          printf("3 \n");
      }
      else{
        angle = 2*M_PI + atan2(deltaY, deltaX); //Fourth Quadrant
          printf("4 \n");
      }

  		printf("Distance: %lf \n"
  			"Angle: %lf \n", Distance, angle);

    }
  }


  double deltaMidX = midPnt.x - prevMidPnt.x;
  double deltaMidY = midPnt.y - prevMidPnt.y;

  double x_movement = x_axisUnit*(double)midPnt.x;

  if(x_movement < 1.7 and x_movement > 1.9){
    cam.move_x_axis(x_movement);
  }


}



void Robotics()
{

  Camera cam(1.8, 1, 14, 1.5);

  //Where the midpoint of the Contour will be hold in;
  int MidX = 0;
  int MidY = 0;

	//Creating Memory For Sequence for Contours;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* FirstContour;
	CvSeq* LoopingContour;
	CvSeq* MatchingContour;

	Input = cvCaptureFromFile(1);//cvCreateCameraCapture("/dev/stdin");//finds the input device
	IplImage* HSV;
	frame = cvQueryFrame(Input); //reading frame from input

  double x_axis_unit = 3 / (double)(frame->width);  //Cartesian X-axis unit
  double y_axis_unit = 3/(double)frame->height;  //Cartesian Y-Axis unit


	//Colour for Contour
	OuterColour = cvScalar(225,0,0,0);
	InnerColour = cvScalar(0,225,0,0);

	//Line Colour
	CvScalar LineColour = cvScalar(0, 0, 225, 0);
  CvScalar AxisColour = cvScalar(225, 0, 0, 0);

	//MidPoint of Contour
	CvPoint MidPnt;

	//Point of Reference/origin
	CvPoint Origin = cvPoint( frame->width/2, frame->height/2);
  printf("Press 49 till 53 to open the various windows. Upon opening a new one will close the current viewed window\nPress Escapte to Exit\n");
	printf("Origin points: width: %d, height: %d\n",Origin.x, Origin.y );

  //points to draw cartesian axis;
  CvPoint y_1 = cvPoint(frame->width/2, frame->height);
  CvPoint y_2 = cvPoint(frame->width/2, 0);

  //points to draw cartesian axis;
  CvPoint x_1 = cvPoint(0, frame->height/2);
  CvPoint x_2 = cvPoint(frame->width, frame->height/2);

  //Point where the midpoint was (dt) ago
  CvPoint prevMidPnt = cvPoint(0,0);

  //Container for axis;
  CvScalar y_axis = cvScalar(frame->width/2, frame->height, frame->width/2, 0 );
  CvScalar x_axis = cvScalar(0, frame->height/2, frame->width, frame->height/2 );


	//Area of Contour;
	double AreaOfContour1 = 0;
	double AreaOfContour2 = 0;


	//User Input
	char c;
	char k;


	//Windows
	cvNamedWindow("Output", 0);
	cvNamedWindow("RGB_Alter", 0);
	cvNamedWindow("GRAY", 0);
	cvNamedWindow("Threshold", 0);
	cvNamedWindow("SearchingForObject", 0);
	cvNamedWindow("Erode", 0);
	cvNamedWindow("Dilate", 0);

	//RGB setting value Trackbar
	cvCreateTrackbar("R-Lower", "SearchingForObject", &R_lower, 255, SetLowerRValue );
	cvCreateTrackbar("G-Lower", "SearchingForObject", &G_lower, 255, SetLowerGValue );
	cvCreateTrackbar("B-Lower", "SearchingForObject", &B_lower, 255, SetLowerBValue );
  // RGB setting value Trackbar
	cvCreateTrackbar("R-Higher", "SearchingForObject", &R_higher, 255, SetHigherRValue );
	cvCreateTrackbar("G-Higher", "SearchingForObject", &G_higher, 255, SetHigherGValue );
	cvCreateTrackbar("B-Higher", "SearchingForObject", &B_higher, 255, SetHigherBValue );
	//thresholding Trackbar
	cvCreateTrackbar("LowerThreshold", "Threshold", &Threshold, 225, SetThreshold );
	cvCreateTrackbar("MaxThreshold", "Threshold", &max_value, 225, SetThresholdMaxValue);

	//Trackbar for Number of Iterations of Dilations/Erosion
	cvCreateTrackbar("ErodeIterations", "Erode", &ErodeIterations, 10, SetNumOfErodeIteration);
	cvCreateTrackbar("DilateIterations", "Dilate", &DilateIterations, 10, SetNumOfDilateIteration);


	while(1)
	{

		frame = cvQueryFrame(Input);//request frame from input device
		DetectedObject = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1); //create blank images to hold frame with certain properties
		HSV = cvCloneImage(frame);  //Clone frame to pointer
		cvCvtColor(frame, HSV, CV_BGR2HSV); //create image to hold frame with certain properties


		cvInRangeS(HSV, LowerBound, UpperBound, DetectedObject);



		switch(k){
			case 50:
        cvShowImage("SearchingForObject", DetectedObject);
				break;
			case 51:
				cvShowImage("Erode", DetectedObject);
				break;
			case 52:
				cvShowImage("Dilate", DetectedObject);
				break;
			case 53: //newly added: erode, threshold, RGB
				cvDestroyWindow("Dilate");
        cvDestroyWindow("Erode");
        cvDestroyWindow("Threshold");
        cvDestroyWindow("RGB_Alter");
        cvDestroyWindow("GRAY");
        cvDestroyWindow("SearchingForObjet");
				break;
		}

		cvErode(DetectedObject, DetectedObject, NULL, ErodeIterations);

    //		cvShowImage("Erode", DetectedObject);

		cvDilate(DetectedObject, DetectedObject, NULL, DilateIterations);

  //		cvShowImage("Dilate", DetectedObject);

		cvFindContours(DetectedObject, storage, &FirstContour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

		// cvDrawContours(frame, FirstContour, OuterColour, InnerColour, 0, 8, 8, cvPoint(0,0));

		if( FirstContour != NULL && k == 54)
		{
			AreaOfContour2 = 0;
			for(LoopingContour = FirstContour; LoopingContour != NULL; LoopingContour = LoopingContour->h_next)
			{
				AreaOfContour1 = cvContourArea(LoopingContour, CV_WHOLE_SEQ, 0);
				if(AreaOfContour1 > AreaOfContour2){ //determine the biggest contour area of al contours on the screen.
					AreaOfContour2 = AreaOfContour1;
					MatchingContour = LoopingContour;
				}
        if(DEBUG){
		        printf("%lf\n", AreaOfContour1 );
        }
			}
      //drae the contour with the biggest area on screen
			cvDrawContours(frame, MatchingContour, OuterColour, InnerColour, 1, 8, 8, cvPoint(0,0));

			CvRect rect = cvBoundingRect(MatchingContour, 1);
			// cvRectangle(frame, cvPoint(rect.x, rect.y), cvPoint(rect.width + rect.x, rect.height + rect.y), InnerColour, 4, 8, 0);
			//changes were made in the above line

      //Previous Midpoint
      prevMidPnt = cvPoint(MidX, MidY);


			//Determining MidPoint of Rectangle
			MidX = (rect.x + rect.width + rect.x)/2;
			MidY = (rect.y + rect.height + rect.y)/2;

			MidPnt = cvPoint(MidX, MidY);
			cvRectangle(frame, MidPnt, MidPnt, OuterColour, 4, 8, 0);//draw ractangle

      //Line of origin to midpoint
			cvLine(frame, Origin, MidPnt, LineColour, 5, 8, 0);

      if(DEBUG){
        //Drawing Cartesian axis;
        cvLine(frame, x_1, x_2, AxisColour, 5, 8, 0);
        cvLine(frame, y_1, y_2, AxisColour, 8, 8, 0);
      }

			CalcAngleAndDist(Origin, MidPnt, prevMidPnt, x_axis_unit, y_axis_unit, cam);
		}

		cvShowImage("Output", frame);

		cvReleaseImage(&HSV);
		// cvReleaseImage(&Gray);
		// cvReleaseImage(&RGB_alter);
		cvReleaseImage(&DetectedObject);

		c = cvWaitKey(33); //get the property for this
		if(c != -1 && c != 255){
			k = c;
			printf("\nThe key pressed: %d \n", c );
		}

		if(c == 27){
			break;
		}

	}


	cvDestroyAllWindows();

	cvReleaseImage(&frame);
	cvReleaseImage(&Gray);
	cvReleaseImage(&RGB_alter);
	cvReleaseImage(&DetectedObject);

}

void Camera::move_x_axis(double xDistance){
  if(xDistance < 1.8 and x_axis > 1.2){  //object to the right
      //pwmWrite(PIN_BASE + pin_x, calcTicks(x_axis, HERTZ));
      x_axis += 0.01;
  }
    else if(xDistance > 1.8 and x_axis < 2.5){//object to the left
      //pwmWrite(PIN_BASE + pin_x, calcTicks(x_axis, HERTZ));
      x_axis -= 0.01;
    }

  }



void Camera::move_y_axis(double yDistance){

    //pwmWrite(PIN_BASE + pin_y, calcTicks(yDistance, HERTZ));


}





int main(int argc, char const *argv[]) {
  Robotics();
  return 0;
}
