#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>

class YOLOV5
{
public:
	YOLOV5(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	float confThreshold;
	float nmsThreshold;

	int numproposal;
	int nout;

	std::vector<std::string> class_names;

	torch::jit::script::Module module;
	torch::DeviceType device_type;

};

YOLOV5::YOLOV5(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->numproposal = 25200;
	this->nout = 85;

	std::ifstream ifs("coco.names");
	std::string line;
	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

	this->module = torch::jit::load(modelpath);
	this->device_type = at::kCUDA;
	this->module.to(this->device_type);
}

void YOLOV5::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(255,0,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);

}


cv::Mat YOLOV5::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;
	int srcw = img.cols, srch = img.rows;
	cv::Mat timg;

	if(srcw != srch)
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

void YOLOV5::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat timg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	cv::Mat blob = cv::dnn::blobFromImage(timg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	auto imgTensor = torch::from_blob(blob.data, {1, 3, this->inpHeight, this->inpWidth});
	std::vector<torch::IValue> inputs;

	inputs.emplace_back(imgTensor.to(this->device_type));

	torch::IValue outputs = this->module.forward(inputs);
	torch::Tensor outtensor = outputs.toTensor().to(at::kCPU);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	float* pdata = outtensor[0].squeeze(0).data_ptr<float>();

	for(int n = 0; n < this->numproposal; ++n)
	{
		float conf = pdata[4];
		if(conf > this->confThreshold)
		{
			float maxss = 0.0;
			int idp = 0;
			for(int k = 0; k < this->nout - 5; ++k)
			{
				if(pdata[k + 5] > maxss)
				{
					maxss = pdata[k + 5];
					idp = k;
				
				}
			
			}
			maxss *= conf;
			if(maxss >= this->confThreshold)
			{
				float cx = (pdata[0] - padw) * ratiow;
				float cy = (pdata[1] - padh) * ratioh;
				float w = pdata[2] * ratiow;
				float h = pdata[3] * ratioh;

				int left = int(cx - 0.5 * w);
				int top = int(cy - 0.5 * h);

				boxes.push_back(cv::Rect(left, top, int(w), int(h)));
				confs.push_back(maxss);
				ids.push_back(idp);
			
			}
		
		}
		pdata += this->nout;
	
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);
	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);

}


int main()
{
	YOLOV5 net(0.7, 0.8, "weights/yolov5s.torchscript.pt");

	cv::Mat srcimg = cv::imread("imgs/person.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/person_libtorch_cpp.jpg", srcimg);

	return 0;

}


