#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <typeinfo>


static const float anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
static const float stride[3] = { 8.0, 16.0, 32.0 };



static inline float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }



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

	// int numproposal;
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
	// this->numproposal = 25200;
	this->nout = 6;


	// std::ifstream ifs("coco.names");
	// std::string line;
	// while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));
	
	this->class_names = {"fire"};

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

	// torch::IValue outputs = this->module.forward(inputs);
	// torch::Tensor outtensor = outputs.toTensor().to(at::kCPU);
	
	auto outputs = this->module.forward(inputs).toList();
	
	// std::cout << outputs.size() << std::endl;
	// torch::Tensor out1 = outputs.get(0).toTensor().to(at::kCPU);
	// torch::Tensor out2 = outputs.get(1).toTensor().to(at::kCPU);
	// torch::Tensor out3 = outputs.get(2).toTensor().to(at::kCPU);	



	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;
	// float* pdata = outtensor[0].squeeze(0).data_ptr<float>();
	int n = 0, q = 0, i = 0, j = 0, c = 0;
	for(n = 0; n < 3; ++n)
	{
		int num_grid_x = (int)(this->inpWidth / stride[n]);
		int num_grid_y = (int)(this->inpHeight / stride[n]);
		int area = num_grid_x * num_grid_y;
		torch::Tensor outtensor = outputs.get(n).toTensor().to(at::kCPU);

		for(q = 0; q < 3; ++q)
		{
			const float anchor_w = anchors[n][q * 2];
			const float anchor_h = anchors[n][q * 2 + 1];
			float *pdata = outtensor.data_ptr<float>() + q * this->nout * area;
			
				
			for(i = 0; i < num_grid_y; ++i)
			{	
				for(j = 0; j < num_grid_x; ++j)
				{
							
					float conf = sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout + 4]);
					if(conf > this->confThreshold)
					{
						float maxss = 0.0;
						int idp = 0;
						for(c = 0; c < this->class_names.size(); ++c)
						{
							float cla_s = sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout + c + 5]);
							if(cla_s > maxss)
							{
								maxss = cla_s;
								idp = c;
							}
						
						}
						maxss *= conf;	
						if(maxss >= this->confThreshold)
						{
							float scx = (sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout]) * 2.0 - 0.5 + j) * stride[n];
							float scy = (sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout + 1]) * 2.0 - 0.5 + i) * stride[n];
							float sw = powf(sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout + 2]) * 2.0, 2.0) * anchor_w;
							float sh = powf(sigmoid(pdata[i * num_grid_x * this->nout + j * this->nout + 3]) * 2.0, 2.0) * anchor_h;
							
							float cx = (scx - padw) * ratiow;
							float cy = (scy - padh) * ratioh;
							float w = sw * ratiow;
							float h = sh * ratioh;

							int left = int(cx - 0.5 * w);
							int top = int(cy - 0.5 * h);

							boxes.push_back(cv::Rect(left, top, int(w), int(h)));
							confs.push_back(maxss);
							ids.push_back(idp);
						}
					}
					
						
				}
					
			}
				


		}

	
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);
	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);
	

}


int main()
{
	YOLOV5 net(0.7, 0.8, "models/lxfire_base_torchscript3out.pt");

	cv::Mat srcimg = cv::imread("imgs/00007.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/00007_cpp_3out.jpg", srcimg);

	return 0;

}


