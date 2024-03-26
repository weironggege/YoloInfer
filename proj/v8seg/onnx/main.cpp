#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

class YOLOV8SEG
{
public:
	YOLOV8SEG(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	void normalize_(cv::Mat img);
	void maskimg2pred(cv::Mat& frame, cv::Mat maskpreds, std::vector<cv::Rect> npboxes, float* maskout);
	void detect(cv::Mat& frame);
private:

	int inpWidth;
	int inpHeight;
	int nm;
	int numproposal;
	int nout;


	float confThreshold;
	float nmsThreshold;

	int mask_width;
	int mask_height;

	std::vector<float> input_image_;
	std::vector<std::string> class_names;


	std::vector<const char*> input_names{"images"};
	std::vector<const char*> output_names{"output0", "output1"};

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV8SEG");

	Ort::Session *sess = nullptr;
	Ort::SessionOptions sessops = Ort::SessionOptions();

};

YOLOV8SEG::YOLOV8SEG(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->nm = 32;
	this->numproposal = 8400;
	this->nout = 116;

	this->mask_width = 160;
	this->mask_height = 160;

	std::ifstream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

	OrtSessionOptionsAppendExecutionProvider_CUDA(this->sessops, 0);

	this->sessops.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	this->sess = new Ort::Session(this->env, modelpath.c_str(), this->sessops);

}

void YOLOV8SEG::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,144,255), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x,box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(144,0,255), 2);


}


void YOLOV8SEG::normalize_(cv::Mat img)
{
	int srch = img.rows, srcw = img.cols;
	this->input_image_.resize(srcw * srch * img.channels());

	for(int c = 0; c < 3; ++c)
	{
		for(int i = 0; i < srch; ++i)
		{
			for(int j = 0; j < srcw; ++j)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c*srcw*srch+i*srcw+j] = pix / 255.0;
			
			}
		}
	
	}

}


void YOLOV8SEG::maskimg2pred(cv::Mat& frame, cv::Mat maskpreds, std::vector<cv::Rect> npboxes, float* maskout)
{
	cv::Mat protos = cv::Mat(this->nm, this->mask_width*this->mask_height, CV_32F, maskout);
	cv::Mat maskRes = (maskpreds * protos).t();
	cv::Mat masks = maskRes.reshape(npboxes.size(), {this->mask_height, this->mask_width});

	float mratiow = (float)frame.cols / this->mask_width;
	float mratioh = (float)frame.rows / this->mask_height;

	cv::Size blur_size = cv::Size(int(mratiow), int(mratioh));
	cv::Mat mask_img = frame.clone();

	cv::RNG rng(456);

	std::vector<cv::Mat> maskChannels;
	cv::split(masks, maskChannels);

	for(int i = 0; i < npboxes.size(); ++i)
	{
		int ox = npboxes[i].x, oy = npboxes[i].y, ow = npboxes[i].width, oh = npboxes[i].height;
		int sx = int(ox / mratiow), sy = int(oy / mratioh), sw = int(ow / mratiow), sh = int(oh / mratioh);

		cv::Mat dest;
		cv::exp(-maskChannels[i], dest);
		dest = 1 / (1 + dest);

		cv::Rect roi(sx, sy, sw, sh);
		cv::Mat scale_crop_mask = dest(roi);

		cv::Mat crop_mask;
		cv::resize(scale_crop_mask, crop_mask, cv::Size(ow, oh), cv::INTER_CUBIC);

		cv::blur(crop_mask, crop_mask, blur_size);

		crop_mask = crop_mask > 0.5;

		int b = rng.uniform(0, 255), g = rng.uniform(0, 255), r = rng.uniform(0, 255);
		mask_img(npboxes[i]).setTo(cv::Scalar(b,g,r), crop_mask);
	
	}

	cv::addWeighted(mask_img, 0.5, frame, 0.5, 0, frame);


}

void YOLOV8SEG::detect(cv::Mat& frame)
{
	cv::Mat dsimg;
	cv::resize(frame, dsimg, cv::Size(this->inpWidth, this->inpHeight));

	this->normalize_(dsimg);

	std::array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	std::vector<Ort::Value> output_tensors = this->sess->Run(Ort::RunOptions{nullptr}, &input_names[0], &input_tensor, 1, output_names.data(), output_names.size());

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Mat> mpreds;
	std::vector<cv::Rect> boxes;

	float ratiow = (float)frame.cols / this->inpWidth;
	float ratioh = (float)frame.rows / this->inpHeight;

	float* pred = output_tensors[0].GetTensorMutableData<float>();

	cv::Mat rawData = cv::Mat(cv::Size(this->numproposal, this->nout), CV_32F, pred).t();

	float* pdata = (float*)rawData.data;

	for(int n = 0; n < this->numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < this->nout-4-this->nm; ++k)
		{
			if(pdata[k + 4] > maxss)
			{
				maxss = pdata[k + 4];
				idp = k;
			
			}
		
		}
		if(maxss >= this->confThreshold)
		{
		
			float cx = pdata[0] * ratiow;
			float cy = pdata[1] * ratioh;
			float w = pdata[2] * ratiow;
			float h = pdata[3] * ratioh;

			int left = int(cx - 0.5 * w);
			int top = int(cy - 0.5 * h);

			boxes.push_back(cv::Rect(left, top, int(w), int(h)));
			confs.push_back(maxss);
			ids.push_back(idp);
			mpreds.push_back(rawData.row(n).colRange(this->nout-this->nm, this->nout));
		}
		pdata += this->nout;
	
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);
	
	std::vector<cv::Rect> npboxes;
	cv::Mat maskpreds;
	float* maskout = output_tensors[1].GetTensorMutableData<float>();

	for(int idx : indices)
	{
		mpreds[idx].convertTo(mpreds[idx], CV_32F);
		npboxes.push_back(boxes[idx]);
		maskpreds.push_back(mpreds[idx]);
	
	}
	this->maskimg2pred(frame, maskpreds, npboxes, maskout);

	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);


}


int main()
{
	YOLOV8SEG net(0.7, 0.8, "weights/yolov8n-seg.onnx");

	cv::Mat srcimg = cv::imread("imgs/person.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/person_onnx_cpp.jpg", srcimg);

	return 0;
}










