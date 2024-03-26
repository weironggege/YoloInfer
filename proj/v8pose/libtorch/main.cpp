#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>

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

	int numproposal;
	int nout;
	int keyPnums;

	float confThreshold;
	float nmsThreshold;

	std::vector<int> skeletons;

	torch::jit::script::Module module;
	torch::DeviceType device_type;

};


YOLOV8POSE::YOLOV8POSE(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;

	this->numproposal = 8400;
	this->nout = 56;
	this->keyPnums = 17;

	std::ifstream ifs("pindex.txt");
	std::string line;

	while(getline(ifs, line)) this->skeletons.push_back(atoi(line.c_str()));

	this->module = torch::jit::load(modelpath);
	this->device_type = at::kCUDA;
	this->module.to(this->device_type);

}

void YOLOV8POSE::drawPred(cv::Mat& frame, cv::Rect box, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0, 144,114), 2);

	std::string label = cv::format("%.2f", conf);

	label = "Person:" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);


}


void YOLOV8POSE::drawPose(cv::Mat& frame, std::vector<float> kps)
{
	cv::RNG rng;

	for(int i = 0; i < this->keyPnums; ++i)
	{
		int b = rng.uniform(0, 255), g = rng.uniform(0, 255), r = rng.uniform(0,255);

		cv::circle(frame, cv::Point(int(kps[i*3]), int(kps[i*3+1])), 5, cv::Scalar(b,g,r), 2);
	
	}

	for(int j = 0; j < this->skeletons.size(); j += 2)
	{
		int b = rng.uniform(0, 255), g = rng.uniform(0, 255), r = rng.uniform(0, 255);
		cv::Point pos1(int(kps[(this->skeletons[j]-1)*3]), int(kps[(this->skeletons[j]-1)*3+1]));
		cv::Point pos2(int(kps[(this->skeletons[j+1]-1)*3]), int(kps[(this->skeletons[j+1]-1)*3+1]));
		cv::line(frame, pos1, pos2, cv::Scalar(b,g,r), 1, cv::LINE_AA);
	
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
			cv::copyMakeBorder(timg, timg, *padh, this->inpHeight-*newh-*padw, 0, 0, cv::BORDER_CONSTANT, 114);
		
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

	torch::Tensor imgTensor = torch::from_blob(blob.data, {1, 3, this->inpHeight, this->inpWidth});

	std::vector<torch::jit::IValue> inputs;

	inputs.emplace_back(imgTensor.to(this->device_type));

	torch::jit::IValue outt = this->module.forward(inputs);

	torch::Tensor outtensor = outt.toTensor().to(at::kCPU).squeeze(0);

	std::vector<float> confs;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> kpss;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;

	cv::Mat rawData = cv::Mat(cv::Size(this->numproposal, this->nout), CV_32F, outtensor.data_ptr<float>()).t();
	float* pdata = (float*)rawData.data;

	for(int n = 0; n < this->numproposal; ++n)
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

			confs.push_back(conf);
			boxes.push_back(cv::Rect(left, top, int(w), int(h)));

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
		pdata += this->nout;
	
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
	YOLOV8POSE net(0.7, 0.8, "weights/yolov8n-pose.torchscript");

	cv::Mat srcimg = cv::imread("imgs/ppose.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/ppose_libtorch_cpp.jpg", srcimg);

	return 0;
}












