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

#include "Car.h"

using namespace std;
using namespace cv;


int main()
{
	Car carObj;
	char config[] = "test.txt";
	char videoFile[]="test.ts";
	if(carObj.ReadConfig(config)<0)
	{
		printf("配置文件错误！\n");
		return 0;
	}
	VideoCapture capture;
	capture.open(videoFile);
	if (!capture.isOpened())
	{
		printf("视频路径不存在");
		return 0;
	}
	while (capture.read(carObj.frame))
	{
		carObj.ProcessOpencv(carObj.frame);
	}

	return 0;
}

