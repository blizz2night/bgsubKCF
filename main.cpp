#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
//#include <fstream>
//#include <sstream>
#include <string>
#include <vector>
#include <list>

//#include <Windows.h>
//#include  <mmsystem.h>
//#pragma comment(lib,"winmm.lib")
#include "kcftracker.hpp"
//#include <time.h>

using namespace std;
using namespace cv;

float bbOverlap(Rect& box1, Rect& box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}
bool isIntersected(Rect& box1, Rect& box2)
{
	if (box1.x > box2.x + box2.width) { return false; }
	if (box1.y > box2.y + box2.height) { return false; }
	if (box1.x + box1.width < box2.x) { return false; }
	if (box1.y + box1.height < box2.y) { return false; }
	return true;
	
}
int main(int argc, char* argv[])
{
	//if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = true;
	bool MULTISCALE = false;
	bool SILENT = true;
	bool LAB = true;
	bool detectShadows = false;
	//for (int i = 0; i < argc; i++){
	//	if (strcmp(argv[i], "hog") == 0)
	//		HOG = true;
	//	if (strcmp(argv[i], "fixed_window") == 0)
	//		FIXEDWINDOW = true;
	//	if (strcmp(argv[i], "singlescale") == 0)
	//		MULTISCALE = false;
	//	if (strcmp(argv[i], "show") == 0)
	//		SILENT = false;
	//	if (strcmp(argv[i], "lab") == 0){
	//		LAB = true;
	//		HOG = true;
	//	}
	//	if (strcmp(argv[i], "gray") == 0)
	//		HOG = false;
	//}

	// Create KCFTracker object
	//KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2();
	
	// Frame readed
	Mat frame;
	Mat gray;
	Mat fg;
	vector<vector<cv::Point>> contours;
	vector<Rect>rects;
	list<KCFTracker> trackers;
	
	vector<Rect>results;
	// Tracker results
	Rect result;
	Mat kernel = Mat::ones(Size(3, 3), CV_8U);
	Point anchor(-1, -1);
	int iteration = 2;
	int fnum = 1;
	int size=0;
	double ratio=0.002;
	int threshold_min = 0;
	int threshold_max = 0;
	int x, y, width, height;
	VideoCapture cap("C:\\Users\\yulie\\Documents\\Visual Studio 2013\\Projects\\kcfTracker\\bgsubKCF\\1.avi");
	//char ch;
	//ifs >> x1;
	//ifs >> ch;
	//ifs >> y1;
	//ifs >> ch;
	//ifs >> width;
	//ifs >> ch;
	//ifs >> height;
	//stringstream sfps;
	//LARGE_INTEGER tt1, tt2, tc;
	//QueryPerformanceFrequency(&tc);
	
	//time_t timeBegin, timeEnd;
	//timeBegin = time(NULL);
	// 1. 定义HOG对象
	cv::HOGDescriptor hog; // 采用默认参数


	// 2. 设置SVM分类器
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());   // 采用已经训练好的行人检测分类器

	// 3. 在测试图像上检测行人区域
	vector<cv::Rect> regions;
	//hog.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1);
	//hog.detectMultiScale(image, regions, 0);

	if (!cap.isOpened())
	{
		cout << "open video failed" << endl;

	}
	//QueryPerformanceCounter(&tt2);
	while (true)
	{
		cap >> frame;
		if (frame.empty())
		{
			break;
		}
		if (fnum==1)
		{
			size = frame.cols*frame.rows;
			threshold_min = size*ratio;
			threshold_max = size*0.5;
			fnum++;
		}
		cvtColor(frame, gray, CV_BGR2GRAY);
		pMOG2->apply(gray, fg);
		morphologyEx(fg, fg, MORPH_OPEN, kernel, anchor, iteration);
		dilate(fg, fg, kernel, anchor, iteration);
		Mat temp;
		fg.copyTo(temp);
		findContours(temp, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		//results.clear();
		//for (list<KCFTracker>::iterator iter = trackers.begin(); iter != trackers.end() ; iter++)
		//{
		//	result = iter->update(frame);
		//	//销毁跟踪器条件
		//	if (result.x>0&&result.y>0)
		//	{
		//		results.push_back(result);
		//		rectangle(frame, result, Scalar(0, 255, 0), 2);
		//	}
		//	else
		//	{
		//		trackers.erase(iter);
		//	}
		//	
		//}

		
		for each (vector<cv::Point> contour in contours)
		{
			int contourSize = contourArea(contour);
			if (contourSize > threshold_min&& contourSize<threshold_max)
			{
				Rect rect = boundingRect(contour);
					vector<Point> locations;
					vector<double> weights;
					Mat roi = frame(rect).clone();
					resize(roi, roi, Size(64, 128));
					hog.detect(roi, locations);
					if (locations.size()>0)
					{
						rectangle(frame, rect, Scalar(0, 0, 255));

					}
				
					//hog.detectROI(frame,result)
					//if (!isIntersected(rect, result))
					//{
					//	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
					//	tracker.init(rect, frame);
					//	trackers.push_back(tracker);
					//}
				

				
			}

		}

		imshow("fg", fg);
		imshow("test", frame);
		
		if (waitKey(5)==27)
		{
			break;
		}

	}
}