#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

class YOLOV8POSE
{
public:
	YOLOV8POSE(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, float conf);
	void drawPose(cv::Mat& frame, std::vector<float> kps);
	void normalize_(cv::Mat img);
	cv::Mat resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh);
	void detect(cv::Mat& frame);
private:

	int inpWidth;
	int inpHeight;
	int keyPnums;

	int numproposal;
	int nout;

	float confThreshold;
	float nmsThreshold;

	std::vector<int> skeletons;
	std::vector<float> input_image_;

	std::vector<const char*> input_names{"images"};
	std::vector<const char*> output_names{"output0"};

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV8POSE");
	Ort::Session *sess = nullptr;
	Ort::SessionOptions sessops = Ort::SessionOptions();
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

	OrtSessionOptionsAppendExecutionProvider_CUDA(this->sessops, 0);
	this->sessops.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	this->sess = new Ort::Session(this->env, modelpath.c_str(), this->sessops);

}

void YOLOV8POSE::drawPred(cv::Mat& frame, cv::Rect box, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

	std::string label = cv::format("%.2f", conf);

	label = "Person:" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,0,0), 2);

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
		if(kps[(this->skeletons[j]-1)*3+2] > 0.5)
		{
			cv::Point pos1(int(kps[(this->skeletons[j]-1)*3]), int(kps[(this->skeletons[j]-1)*3+1]));
			cv::Point pos2(int(kps[(this->skeletons[j+1]-1)*3]), int(kps[(this->skeletons[j+1]-1)*3+1]));
			cv::line(frame, pos1, pos2, cv::Scalar(b, g, r), 1, cv::LINE_AA);
		
		}
	}

}


void YOLOV8POSE::normalize_(cv::Mat img)
{
	int srch = img.rows, srcw = img.cols;
	this->input_image_.resize(srch * srcw * img.channels());

	for(int c = 0; c < 3; ++c)
	{
		for(int i = 0; i < srch; ++i)
		{
			for(int j = 0; j < srcw; ++j)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * srcw * srch + i * srcw + j] = pix / 255.0;
			
			}
		}
	}


}


cv::Mat YOLOV8POSE::resize_img(cv::Mat img, int *neww, int *newh, int *padw, int *padh)
{
	*neww = this->inpWidth, *newh = this->inpHeight;
	int srcw = img.cols, srch = img.rows;
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
	this->normalize_(simg);

	std::array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	std::vector<Ort::Value> output_tensors = this->sess->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor, 1, output_names.data(), output_names.size());

	std::vector<float> confs;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> kpss;

	float ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / newh;

	auto output_ptr = output_tensors[0].GetTensorMutableData<float>();

	cv::Mat rawData = cv::Mat(cv::Size(this->numproposal, this->nout), CV_32F, output_ptr).t();
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
	YOLOV8POSE net(0.7, 0.8, "weights/yolov8n-pose.onnx");

	cv::Mat srcimg = cv::imread("imgs/ppose.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/ppose_onnx_cpp.jpg", srcimg);

	return 0;
}




