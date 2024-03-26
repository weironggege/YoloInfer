#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>

class YOLOV8SEG
{
public:
	YOLOV8SEG(float confThreshold, float nmsThreshold, std::string modelpath);
	void drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf);
	void maskimg2Pred(cv::Mat& frame, cv::Mat mask_preds, std::vector<cv::Rect> np_boxes, cv::Mat maskout);
	void detect(cv::Mat& frame);
private:
	int inpWidth;
	int inpHeight;

	int nm;
	float confThreshold;
	float nmsThreshold;

	cv::dnn::Net net;

	std::vector<std::string> class_names;
};



YOLOV8SEG::YOLOV8SEG(float confThreshold, float nmsThreshold, std::string modelpath)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;

	this->inpWidth = 640;
	this->inpHeight = 640;
	this->nm = 32;

	this->net = cv::dnn::readNet(modelpath);

	std::ifstream ifs("coco.names");
	std::string line;

	while(getline(ifs, line)) this->class_names.push_back(line.substr(0, line.length()-1));

}

void YOLOV8SEG::drawPred(cv::Mat& frame, cv::Rect box, int classid, float conf)
{
	cv::rectangle(frame, box, cv::Scalar(0,255,144), 2);

	std::string label = cv::format("%.2f", conf);

	label = this->class_names[classid] + ":" + label;

	cv::putText(frame, label, cv::Point(box.x, box.y-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(144,255,0), 2);


}

void YOLOV8SEG::maskimg2Pred(cv::Mat& frame, cv::Mat mask_preds, std::vector<cv::Rect> np_boxes, cv::Mat maskout)
{
	int num_mask = maskout.size[1], mask_height = maskout.size[2], mask_width = maskout.size[3];
	float* maskptr = (float*)maskout.data;
	cv::Mat protos = cv::Mat(num_mask, mask_height*mask_width, CV_32F, maskptr);
	cv::Mat maskRes = (mask_preds * protos).t();
	cv::Mat masks = maskRes.reshape(np_boxes.size(), {mask_height, mask_width});

	float mbratiow = (float)frame.cols / mask_width, mbratioh = (float)frame.rows / mask_height;
	cv::Size blur_size = cv::Size(int(mbratiow), int(mbratioh));
	cv::Mat mask_img = frame.clone();

	std::vector<cv::Mat> maskChannels;
	cv::split(masks, maskChannels);
	cv::RNG rng(345);

	for(int i = 0; i < np_boxes.size(); ++i)
	{
		int ox = np_boxes[i].x, oy = np_boxes[i].y, ow = np_boxes[i].width, oh = np_boxes[i].height;
		int sx = int(ox / mbratiow), sy = int(oy / mbratioh), sw = int(ow / mbratiow), sh = int(oh / mbratioh);

		cv::Mat dest;
		cv::exp(-maskChannels[i], dest);
		dest = 1 / (1 + dest);

		cv::Rect roi(sx, sy, sw, sh);
		cv::Mat scale_crop_mask = dest(roi);

		cv::Mat crop_mask;
		cv::resize(scale_crop_mask, crop_mask, cv::Size(ow, oh), cv::INTER_CUBIC);
		cv::blur(crop_mask, crop_mask, blur_size);

		crop_mask = crop_mask > 0.5;
		
		int b = rng.uniform(0,255), g = rng.uniform(0,255), r = rng.uniform(0, 255);
		mask_img(np_boxes[i]).setTo(cv::Scalar(b,g,r), crop_mask);
	
	}
	cv::addWeighted(mask_img, 0.5, frame, 0.5, 0, frame);

}

void YOLOV8SEG::detect(cv::Mat& frame)
{
	cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(this->inpHeight, this->inpWidth), cv::Scalar(0,0,0), true, false);

	this->net.setInput(blob);

	std::vector<cv::Mat> outs;

	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int numproposal = outs[0].size[2];
	int nout = outs[0].size[1];
	outs[0] = outs[0].reshape(0, nout);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> m_preds;

	float ratiow = (float)frame.cols / this->inpWidth;
	float ratioh = (float)frame.rows / this->inpHeight;
	cv::transpose(outs[0], outs[0]);
	float* pdata = (float*)outs[0].data;

	for(int n = 0; n < numproposal; ++n)
	{
		float maxss = 0.0;
		int idp = 0;
		for(int k = 0; k < nout - 4 - this->nm; ++k)
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
			m_preds.push_back(outs[0].row(n).colRange(nout-this->nm, nout));
		
		}
	
		pdata += nout;
	}
	
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);

	std::vector<cv::Rect> np_boxes;
	cv::Mat mask_preds;

	for(int idx : indices)
	{
		m_preds[idx].convertTo(m_preds[idx], CV_32F);
		np_boxes.push_back(boxes[idx]);
		mask_preds.push_back(m_preds[idx]);
	
	}

	this->maskimg2Pred(frame, mask_preds, np_boxes, outs[1]);

	for(int idx : indices) this->drawPred(frame, boxes[idx], ids[idx], confs[idx]);
}


int main()
{
	YOLOV8SEG net(0.7, 0.8, "weights/yolov8n-seg.onnx");

	cv::Mat srcimg = cv::imread("imgs/person.jpg");

	net.detect(srcimg);

	cv::imwrite("imgs/person_opencv_cpp.jpg", srcimg);

	return 0;


}











