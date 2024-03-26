#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

class YOLOV8POSE
{
public:
	YOLOV8POSE(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, float conf);
	void drawPose(cv::Mat& frame, std::vector<float> kps);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int keyPnums;

	float confThreshold;
	float nmsThreshold;

	std::vector<int> skeletons;

	cv::dnn::Net net;

};

YOLOV8POSE::YOLOV8POSE(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->keyPnums = 17;

	this->net = cv::dnn::readNet(modelpath);

	std::ifstream ifs("pindex.txt");
	std::string line;

	while(getline(ifs, line)) this->skeletons.push_back(atoi(line.c_str()));


}


void YOLOV8POSE::drawPred(cv::Mat& frame, cv::Rect box, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = "Person:" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2);

}

void YOLOV8POSE::drawPose(cv::Mat& frame, std::vector<float> kps)
{
	cv::RNG rng;
	for(int i = 0; i < this->keyPnums; ++i)
	{	
		int b = rng.uniform(0, 255), g = rng.uniform(0, 255), r = rng.uniform(0, 255);
		if(kps[i * 3 + 2] > 0.5)
		{
			cv::circle(frame, cv::Point(int(kps[i*3]), int(kps[i*3+1])), 5, cv::Scalar(b,g,r), 2);
		}
	
	}

	for(int j = 0; j < this->skeletons.size(); j += 2)
	{
		int b = rng.uniform(0, 255), g = rng.uniform(0, 255), r = rng.uniform(0, 255);
		if(kps[(this->skeletons[j]-1) * 3 + 2] > 0.5)
		{
			cv::Point pos1(int(kps[(this->skeletons[j]-1)*3]), int(kps[(this->skeletons[j]-1)*3+1]));
			cv::Point pos2(int(kps[(this->skeletons[j+1]-1)*3]), int(kps[(this->skeletons[j+1]-1)*3+1]));
			cv::line(frame, pos1, pos2, cv::Scalar(r,g,b), 2, cv::LINE_AA);
		}
	
	}


}

cv::Mat YOLOV8POSE::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;
	int srch = img.rows, srcw = img.cols;
	cv::Mat timg;

	if(srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if(hw_scale > 1.0)
		{
			*neww = int(this->inpWidth / hw_scale);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(timg, timg, 0, 0, *padw, this->inpWidth-*neww-*padw, cv::BORDER_CONSTANT, 114);
		
		}
		else
		{
			*newh = int(this->inpHeight * hw_scale);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padh = int((this->inpHeight - *newh) * 0.5);
			cv::copyMakeBorder(timg, timg, *padh, this->inpHeight-*newh-*padh, 0, 0, cv::BORDER_CONSTANT, 114);
		
		}
	
	}
	else
	{
		cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
	
	}
	return timg;
}

void YOLOV8POSE::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat simg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	cv::Mat blob = cv::dnn::blobFromImage(simg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	this->net.setInput(blob);

	std::vector<cv::Mat> outs;

	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int numproposal = outs[0].size[2];
	int nout = outs[0].size[1];

	outs[0] = outs[0].reshape(0, nout);
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;

	std::vector<std::vector<float>> kpss;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	cv::transpose(outs[0], outs[0]);
	float* pdata = (float*)outs[0].data;

	for(int n = 0; n < numproposal; ++n)
	{
		float conf = pdata[4];
		if(conf > this->confThreshold)
		{
			float cx = (pdata[0] - padw) * ratiow;
			float cy = (pdata[1] - padh) * ratioh;
			float w = pdata[2] * ratiow;
			float h = pdata[3] * ratioh;

			int left = int(cx - 0.5 * w);
			int top = int(cy - 0.5 * h);

			boxes.push_back(cv::Rect(left, top, int(w), int(h)));
			confs.push_back(conf);
			
			std::vector<float> kps;

			for(int i = 0; i < this->keyPnums; ++i)
			{
				float kpx = (pdata[i * 3 + 5] - padw) * ratiow;
				float kpy = (pdata[i * 3 + 6] - padh) * ratioh;
				float kpv = pdata[i * 3 + 7];

				kps.push_back(kpx);
				kps.push_back(kpy);
				kps.push_back(kpv);
			
			}
			kpss.push_back(kps);
		}
		pdata += nout;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);

	for(int idx : indices)
	{
		this->drawPred(frame, boxes[idx], confs[idx]);
		this->drawPose(frame, kpss[idx]);
	}

}

int main()
{
	YOLOV8POSE net(0.7, 0.8, "weights/yolov8n-pose.onnx");

	cv::Mat srcimg = cv::imread("imgs/ppose.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/ppose_opencv_cpp.jpg", srcimg);

	return 0;


}



