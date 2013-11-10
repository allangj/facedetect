/******************************************************************************
 *
 * This is the source file for the face detector on an input image
 *
 * Copyright (C) 2010 by Allan Granados Jiménez (allangj1_618@hotmail.com)
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 ******************************************************************************/

/****************************** Include Headers *******************************/
#include <iostream>
#include <stdio.h>
/* Include Headers OpenCV */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
/***************************** Set name spaces ********************************/
using namespace std;
using namespace cv;

/****************************** Global defines ********************************/
/* Scale to resize the output */
#define RESULT_SCALE 1

/******************************* Global variables *****************************/
/* Default location of haarcascade analysis files */
String cascadeName = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml"; // FIXME

/***************************** Function prototypes ****************************/
/* Detect and draw function */
void detectAndDraw(
   Mat &              img, 
   CascadeClassifier &cascade,
   double             scale,
   String             outFile);

/* Main function */
int main(
   int          argc, 
   const char **argv )
{
   Mat image;
   CascadeClassifier cascade; // Cascade to use
   double scale = RESULT_SCALE; // Scale to use
   String inputName; // Name of the input file
   String outputName; // Name of the input file

   if (argc != 3)
   {
      cout << "Usage: ./facedetect <input_file> <output_file>" << endl;
      return -1;
   }

   // Assign input file
   inputName.assign(argv[1]);

   // Load input image
   if(inputName.size())
   {
      image = imread(inputName, 1);
      if(image.empty())
      {
         cout << "Read image didn't work" << endl;
         return -1;
      }
   }
   else
   {
      cout << "No input file" << endl;
      return -1;
   }

   // Assign outfile file
   outputName.assign(argv[2]);

   // If we can load primary cascade notify
   if(!cascade.load(cascadeName))
   {
      cerr << "ERROR: Could not load classifier cascade" << endl;
      return -1;
   }

   if(!image.empty())
   {
      detectAndDraw(image, cascade, scale, outputName);
   }

   return 0;
} /* main() */

/* Function Implementation */
void detectAndDraw(
   Mat&               img, 
   CascadeClassifier &cascade, 
   double             scale,
   String             outFile)
{
   int i = 0;
   double t = 0;
   vector<Rect> faces;
   const static Scalar colors[] =
   { 
      CV_RGB(0,0,255),
      CV_RGB(0,128,255),
      CV_RGB(0,255,255),
      CV_RGB(0,255,0),
      CV_RGB(255,128,0),
      CV_RGB(255,255,0),
      CV_RGB(255,0,0),
      CV_RGB(255,0,255)
   };

   /* Mat for gray image and resized image 
      If we set always resize to 1 we may not use smallImg*/
   Mat gray, smallImg(cvRound(img.rows/scale), cvRound(img.cols/scale), CV_8UC1);

   /* Transform image to gray scale */
   cvtColor(img, gray, CV_BGR2GRAY);
   /* Resize image to the small image MAT */
   resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
   /* Do a equalization on the small image histogram, it is done seeking
      contrast in an image, in order to stretch out the intensity range */
   equalizeHist(smallImg, smallImg );

   /* Get the time so we can measure the process */
   t = (double)cvGetTickCount();

   /* Detects objects of different sizes in the input image. 
      The detected objects are returned as a list of rectangles */
   cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0
        //|CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE, Size(30, 30));

   /* Get the processing time interval */
   t = (double)cvGetTickCount() - t;

   /* Print detection time */
   printf("Time= %g ms\n", t/((double)cvGetTickFrequency()*1000.));

   /* On the detected areas on the vector do nested cascade and create circles */
   for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
   {
      Mat smallImgROI;
      vector<Rect> nestedObjects;
      Point center;
      Scalar color = colors[i%8];
      int radius;

      /* Set cordinates around the detected face */
      center.x = cvRound((r->x + r->width*0.5)*scale);
      center.y = cvRound((r->y + r->height*0.5)*scale);
      radius = cvRound((r->width + r->height)*0.25*scale);

      /* Draw a circle around the detected face */
      circle(img, center, radius, color, 3, 8, 0);
   }
   
   /* Store processed image */
   imwrite(outFile, img);

} /* detectAndDraw() */

