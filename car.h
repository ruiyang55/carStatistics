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

// ����
struct LaneLine{
	Rect detectRoi;
	int laneID;   //�������
	int laneType; //�������ͣ�1��ֱ�У�2����ת��3����ת��4��ֱ����ת��5��ֱ����ת
	BackgroundSubtractorMOG2  mog; // �г���ǰ��ʱ�ż��
	Mat fore;
	float r; // ǰ����ռ����
};

// ���ٽ�ֹ��
struct BroadLine{
	Point begin;
	Point end;
	int broadType;//�߽����� 1��ֱ�б߽磬2����߽磬3���ұ߽�
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