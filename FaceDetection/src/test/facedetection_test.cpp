/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is an example of how to use SeetaFace engine for face detection, the
 * face detection method described in the following paper:
 *
 *
 *   Funnel-structured cascade for multi-view face detection with alignment awareness,
 *   Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
 *   In Neurocomputing (under review)
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Shuzhe Wu (a Ph.D supervised by Prof. Shiguang Shan)
 *
 * As an open-source face recognition engine: you can redistribute SeetaFace source codes
 * and/or modify it under the terms of the BSD 2-Clause License.
 *
 * You should have received a copy of the BSD 2-Clause License along with the software.
 * If not, see < https://opensource.org/licenses/BSD-2-Clause>.
 *
 * Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
 *
 * Note: the above information must be kept whenever or wherever the codes are used.
 *
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\opencv.hpp"

#include "face_detection.h"

using namespace std;

int main(int argc, char** argv) {
	//if (argc < 3) {
	//    cout << "Usage: " << argv[0]
	//        << " image_path model_path"
	//        << endl;
	//    return -1;
	//}

	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FPS, 10);
	const char* model_path = "D:/Program Files/SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin";
	seeta::FaceDetection detector(model_path);
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	cv::Mat img;
	const int pts_num = 5;
	cv::Rect face_rect;
	cv::Mat img_gray;
	seeta::ImageData img_data;
	img_data.num_channels = 1;
	while (true){
		cap >> img;
		cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);
		int32_t num_face = static_cast<int32_t>(faces.size());

		for (int32_t i = 0; i < num_face; i++) {
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;

			cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
		}

		cv::imshow("Test", img);
		cv::waitKey(100);
	}
}
