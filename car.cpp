#include "car.h"
#include "svm.h"

Car::Car()
{
	frameNo = 0;
	// adaboost
	cascade.load("npos_5051neg_24043_1080.xml");
	// hog
	hog = HOGDescriptor(Size(88, 24), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	Model = svm_load_model("licence_pos10616_neg16610.model");
}

Car::~Car()
{

}

int Car::ReadConfig(char path[])
{
	FILE *fp= fopen(path,"r");
	if(fp==NULL)
	{
		cout<<"无法读取配置文件请检查配置文件路径是否正确！\n";
		return -1;
	}

	fscanf(fp,"%d\n",&scale);

	int laneNum=0;
	fscanf(fp,"%d\n",&laneNum);
	for (int i=0;i<laneNum;i++)
	{
		LaneLine ll;
		fscanf(fp,"%d %d %d %d %d %d\n",&ll.detectRoi.x,&ll.detectRoi.y,&ll.detectRoi.width,&ll.detectRoi.height,&ll.laneID,&ll.laneType);
		ll_list.push_back(ll);
	}
	int broadLineNum;
	fscanf(fp, "%d\n", &broadLineNum);
	for (int i=0;i<broadLineNum;i++)
	{
		BroadLine bl;
		fscanf(fp,"%d %d %d %d %d\n",&bl.begin.x,&bl.begin.y,&bl.end.x,&bl.end.y,&bl.broadType);
		bl_list.push_back(bl);
	}
	fclose(fp);
	return 0;
}

void Car::ProcessOpencv(Mat img)
{
	frameNo++;
	if (frameNo % 2 != 0)
		return;

	for(int i=0;i<ll_list.size();i++)
	{
		Rect r = ll_list[i].detectRoi;
		Mat roiImg = img(r);
		DetectCar(roiImg);
	}


	// 显示图像
	for(int i=0;i<ll_list.size();i++)
	{
 		if(i%2==0)
			rectangle(frame,ll_list.at(i).detectRoi, Scalar(255,0,0), 3, 8, 0); 
 		else
			rectangle(frame,ll_list.at(i).detectRoi, Scalar(0,0,255), 3, 8, 0); 
	}
	for(int i=0;i<bl_list.size();i++)
	{
		line(frame,bl_list.at(i).begin,bl_list.at(i).end, Scalar(0,0,255), 3, 8, 0); 
	}

	namedWindow("frame",0);
	imshow("frame", frame);
	int key=waitKey(1); 
	if(key>0) 
		waitKey(0);
}

void Car::DetectCar(Mat img)
{
	vector<Rect> cars; 
	cascade.detectMultiScale(img,cars,1.1,3,0,Size(60,20),Size(300,100));
	for(int j = 0; j < cars.size(); j++)
	{
		Mat carImg = img(cars[j]);
		if(IsLicense(carImg))
			rectangle(img, cars[j], Scalar(0,0,255), 2, 8);
		else
			rectangle(img, cars[j], Scalar(0,255,0), 2, 8);
	}
}

bool Car::IsLicense(Mat img) // 自己完善
{

	vector<float> descriptors;//HOG描述子向量
	//hog.compute();
	bool res = SvmPredict(Model, &descriptors[0], descriptors.size());
	return res;
}

bool Car::SvmPredict(struct svm_model *testModel, float *testArr, int len)
{
	struct svm_node *testX;
	int featureDim = len;
	testX = new struct svm_node[len + 1];
	for (int i = 0; i<featureDim; i++)
	{
		testX[i].index = i;
		testX[i].value = testArr[i];
	}
	testX[featureDim].index = -1;
	float p = svm_predict(testModel, testX);
	delete[] testX;
	if (p > 0.5)
	{
		return true;
	}
	else
	{
		return false;
	}
}