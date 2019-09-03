#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

class FaceDetector{
private:
  const size_t inWidth;
  const size_t inHeight;
  const double inScaleFactor;
  const double confidenceThreshold;
  const cv::Scalar meanVal;
  const std::string configFile;
  const std::string weightFile;
  cv::Mat frame;
  std::vector<cv::Rect> faces;
  cv::dnn::Net net;
public:
  FaceDetector();
  void detectFace(cv::Mat& inFrame);
  std::vector<cv::Rect> getDetectedFaces();
  cv::Mat getProcessedFrame();
};

#endif