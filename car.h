#pragma once 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <string>
#include <iostream>
#include "svm.h"

using namespace cv;
using namespace std;

// 车道
struct LaneLine{
	Rect detectRoi;
	int laneID;   //车道编号
	int laneType; //车道类型，1：直行，2：左转，3：右转，4：直行左转，5：直行右转
	BackgroundSubtractorMOG2  mog; // 有车辆前景时才检测
	Mat fore;
	float r; // 前景所占比例
};

// 跟踪截止线
struct BroadLine{
	Point begin;
	Point end;
	int broadType;//边界类型 1：直行边界，2：左边界，3：右边界
};

class Car{
public:
	Car();
	~Car();
	int ReadConfig(char path[]);
	void ProcessOpencv(Mat img);
	void DetectCar(Mat img);
	bool IsLicense(Mat img);
	bool SvmPredict(struct svm_model *testModel, float *testArr, int len);

public:
	int scale;
	unsigned int frameNo; 
	Mat frame;
	vector<LaneLine> ll_list;
	vector<BroadLine> bl_list;

	CascadeClassifier cascade;  

	HOGDescriptor hog;
	struct svm_model *Model;
};