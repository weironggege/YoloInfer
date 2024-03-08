#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <torch/torch.h>
#include <torch/script.h>

class YOLOV8
{
public:
	YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	int numproposal;
	int nout;

	float confThreshold;
	float nmsThreshold;

	std::vector<std::string> class_names;

	torch::jit::script::Module module;
	torch::DeviceType device_type;

};


YOLOV8::YOLOV8(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->numproposal = 8400;
	this->nout = 84;

	std::ifstream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

	this->module = torch::jit::load(modelpath);
	this->device_type = at::kCUDA;

	this->module.to(this->device_type);

}


void YOLOV8::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);

}

cv::Mat YOLOV8::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;

	int srch = img.rows, srcw = img.cols;
	cv::Mat timg;

	if(srch != srcw)
	{
		float scale_hw = (float)srch / srcw;
		if(scale_hw > 1.0)
		{
			*neww = int(this->inpWidth / scale_hw);
			cv::resize(img, timg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*padw = int((this->inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(timg, timg, 0, 0, *padw, this->inpWidth-*neww-*padw, cv::BORDER_CONSTANT, 114);
		
		}
		else
		{
			*newh = int(this->inpHeight * scale_hw);
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

void YOLOV8::detect(cv::Mat& frame)
{
	int neww = 0, newh = 0, padw = 0, padh = 0;
	cv::Mat simg = this->resize_img(frame, &neww, &newh, &padw, &padh);
	cv::Mat blob = cv::dnn::blobFromImage(simg, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	torch::Tensor imgTensor = torch::from_blob(blob.data, {1, 3, this->inpHeight, this->inpWidth});

	std::vector<torch::jit::IValue> inputs;

	inputs.emplace_back(imgTensor.to(this->device_type));

	torch::jit::IValue outputs = this->module.forward(inputs);

	torch::Tensor out_tensor = outputs.toTensor().to(at::kCPU).squeeze(0);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;

	cv::Mat rawData = cv::Mat(cv::Size(this->numproposal, this->nout), CV_32F, out_tensor.data_ptr<float>()).t();
	float* pdata = (float*)rawData.data;

	for(int n = 0; n < this->numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < this->nout - 4; ++k)
		{
			if(pdata[k + 4] > maxss)
			{
				maxss = pdata[k + 4];
				idp = k;
			}
		
		}
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
		pdata += this->nout;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);

	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);

}

int main()
{
	YOLOV8 net(0.7, 0.8, "weights/yolov8n.torchscript");

	cv::Mat srcimg = cv::imread("imgs/bus.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/bus_libtorch_cpp.jpg", srcimg);

	return 0;


}
