/*
 *
 * This file is part of the open-source SeetaFace engine, which includes three modules:
 * SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
 *
 * This file is part of the SeetaFace Alignment module, containing codes implementing the
 * facial landmarks location method described in the following paper:
 *
 *
 *   Coarse-to-Fine Auto-Encoder Networks (CFAN) for Real-Time Face Alignment, 
 *   Jie Zhang, Shiguang Shan, Meina Kan, Xilin Chen. In Proceeding of the
 *   European Conference on Computer Vision (ECCV), 2014
 *
 *
 * Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
 * Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
 *
 * The codes are mainly developed by Jie Zhang (a Ph.D supervised by Prof. Shiguang Shan)
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

#include "face_alignment.h"

#include <string>
#include <math.h>
#include "cfan.h"
#include "opencv2/opencv.hpp"
namespace seeta {
  /** A constructor with an optional argument specifying path of the model file.0
   *  If called with no argument, the model file is assumed to be stored in the
   *  the working directory as "seeta_fa_v1.1.bin".
   *
   *  @param model_path Path of the model file, either absolute or relative to
   *  the working directory.
   */
  FaceAlignment::FaceAlignment(const char * model_path){
    facial_detector = new CCFAN();
    if (model_path == NULL)
      model_path = "seeta_fa_v1.1.bin";
    facial_detector->InitModel(model_path);
  }

  /** Detect five facial landmarks, i.e., two eye centers, nose tip and two mouth corners.
   *  @param gray_im A grayscale image
   *  @param face_info The face bounding box
   *  @param[out] points The locations of detected facial points
   */
  bool FaceAlignment::PointDetectLandmarks(ImageData gray_im, FaceInfo face_info, FacialLandmark *points)
  {
    if (gray_im.num_channels != 1) {
      return false;
    }
    int pts_num = 5;
    float *facial_loc = new float[pts_num * 2];
    facial_detector->FacialPointLocate(gray_im.data, gray_im.width, gray_im.height, face_info, facial_loc);

    for (int i = 0; i < pts_num; i++) {
      points[i].x = facial_loc[i * 2];
      points[i].y = facial_loc[i * 2 + 1];
    }

    delete[]facial_loc;
    return true;
  }

  /** Pose estimator 
  * @param gray_im The gray image of the data
  * @param points The location of detected facial points: 5 points - 2 eyes, nose, 2 corners of lips
  * @ param[out] face_info the yaw is the needed yaw
  */
  bool FaceAlignment::FacePoseEstimate(const ImageData& gray_im, FacialLandmark* points, FaceInfo& face_info){
	  const double width = gray_im.width;
	  const double height = gray_im.height;
	  const double fov_rad = 60; //if the fov us changed, this need be changed, the default is 60
	  double fov = fov_rad * 3.1416 / 180;
	  cv::Point2d center= cv::Point2d(width / 2, height / 2);
	  double focal_length = center.x / tan(fov);
	  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	  cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double> :: type); //default is (0, 0, 0, 0)

	  // Using the original openCV: t.cn/RYXrVQ7
	  std::vector<cv::Point2d> image_points;
	  // left eye, right eye, nose, left mouth, right mouth
	  image_points.push_back(cv::Point2d((points[0].x) , (points[0].y)));
	  image_points.push_back(cv::Point2d((points[1].x) , (points[1].y)));
	  image_points.push_back(cv::Point2d((points[2].x), (points[2].y)));
	  image_points.push_back(cv::Point2d((points[3].x), (points[3].y)));
	  image_points.push_back(cv::Point2d((points[4].x), (points[4].y)));
		
	  // TODO: model_points⣬http://www.morethantechnical.com/2010/03/19/quick-and-easy-head-pose-estimation-with-opencv-w-code/
	  std::vector<cv::Point3d> model_points;		
	  model_points.push_back(cv::Point3d(2.37427, 110.322, 21.7776));    //l eye
	  model_points.push_back(cv::Point3d(70.0602, 109.898, 20.8234));              //r eye
	  model_points.push_back(cv::Point3d(36.8301, 78.3185, 52.0345)); //nose
	  model_points.push_back(cv::Point3d(14.8498, 51.0115, 30.2378));   //l mouth
	  model_points.push_back(cv::Point3d(58.1825, 51.0115, 29.6224));
	  cv::Mat rotation_vector;
	  cv::Mat translation_vector;
	  cv::Mat rotation_matrix;

	  cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
	  cv::Rodrigues(rotation_vector, rotation_matrix);
	  //
	  //float sy = sqrt(rotation_matrix.at<double>(0, 0)*rotation_matrix.at<double>(0, 0) + rotation_matrix.at<double>(1, 0)*rotation_matrix.at<double>(1, 0));
	  //bool singular = sy < 1e-6;
	  //if (!singular){
		 // face_info.roll = atan2(rotation_matrix.at<double>(2, 1), rotation_matrix.at<double>(2, 2))* 57.2958f;
		 // face_info.yaw = atan2(-rotation_matrix.at<double>(2, 0), sy) * 57.2958f;
		 // face_info.pitch = atan2(rotation_matrix.at<double>(1, 0), rotation_matrix.at<double>(0, 0))* 57.2958f;
	  //}
	  //else{
		 // face_info.roll = atan2(rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(1, 1))* 57.2958f;
		 // face_info.yaw = atan2(-rotation_matrix.at<double>(2, 0), sy)* 57.2958f;
		 // face_info.pitch = 0;
	  //}
	  double theta1 = atan2(rotation_matrix.at<double>(1, 2), rotation_matrix.at<double>(2, 2));
	  double c2 = sqrt(rotation_matrix.at<double>(0, 0)*rotation_matrix.at<double>(0, 0) + rotation_matrix.at<double>(0, 1)*rotation_matrix.at<double>(0, 1));
	  double theta2 = atan2(-rotation_matrix.at<double>(0, 2), c2);
	  double s1 = sin(theta1);
	  double c1 = cos(theta1);
	  double theta3 = atan2(s1*rotation_matrix.at<double>(2, 0) - c1*rotation_matrix.at<double>(1, 0), 
		                                 c1*rotation_matrix.at<double>(1, 1) - s1*rotation_matrix.at<double>(2, 1));
	  face_info.roll = theta1*57.2958f;
	  face_info.yaw = theta2*57.2958f;
	  face_info.pitch = theta3*57.2958f;
	  return true;
  }
  /** A Destructor which should never be called explicitly.
   *  Release all dynamically allocated resources.
   */
  FaceAlignment::~FaceAlignment() {
    if (facial_detector != NULL) {
      delete facial_detector;
      facial_detector = NULL;
    }
  }
}  // namespace seeta
