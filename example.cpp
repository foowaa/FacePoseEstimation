#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"

#include "face_detection.h"
#include "face_alignment.h"
using namespace std;

int main(int argc, char** argv) {
	std::string& _num2str(int num);
    //	cv::VideoCapture cap(0);
	// cap.set(CV_CAP_PROP_FPS, 20);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 250);  
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 250);  
	const char* model_path = "D:/Program Files/SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin";
	const char* model_align_path = "D:/Program Files/SeetaFaceEngine/faceAlignment/model/seeta_fa_v1.1.bin";
	const char* data_path = "D:/Program Files/SeetaFaceEngine/FaceAlignment/data/Angelina_Jolie_0020.jpg";
	seeta::FaceDetection detector(model_path);
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	int pts_num = 5;
	cv::Mat img;
	seeta::FaceAlignment point_detector(model_align_path);
	cv::Rect face_rect;
	cv::Mat img_gray;
	cv::Mat img_gray_resize;
	seeta::ImageData img_data;
	img_data.num_channels = 1;
	//while (true){
    	//cap >> img;
     	img = cv::imread(data_path);
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
		int32_t num_face = static_cast<int32_t>(faces.size());
		if (num_face == 0){
			//TODO: if no faces
		}
		for (int32_t i = 0; i < num_face; i++) {
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;

			cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);

			seeta::FacialLandmark points[5];
			point_detector.PointDetectLandmarks(img_data, faces[0], points);
			for (int j = 0; j < pts_num; j++){
				cv::circle(img, cv::Point(points[j].x, points[j].y), 5, cv::Scalar(0, 255, 0), CV_FILLED);
			}
			point_detector.FacePoseEstimate(img_data, points, faces[0]);
			
			std::ostringstream lines;
			
			lines << "roll: ";
			lines << setprecision(2) << std::fixed << faces[0].roll;
			lines << "//yaw: ";
			lines << setprecision(2) << std::fixed << faces[0].yaw;
			lines << "//pitch: ";
			lines << setprecision(2) << std::fixed << faces[0].pitch;
			std::string& text = lines.str();
			/*cv::putText(img, text, cv::Point(0, img.rows*0.99 ), CV_FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(255, 0, 0));*/
			cv::putText(img, text, cv::Point(0, img.rows*0.99), CV_FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255, 0, 0));
		}

		cv::imshow("Test", img);
		cv::waitKey(100);
	   cv::imwrite("D:/Program Files/SeetaFaceEngine/FaceAlignment/data/Angelina_Jolie_0020_T.jpg", img);
	//}
}


std::string& _num2str(int num){
	std::stringstream ss;
	ss << num;
	return ss.str();
}