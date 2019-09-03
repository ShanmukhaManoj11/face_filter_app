#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

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
  cv::Ptr<cv::face::Facemark> facemark;
  std::vector<std::vector<cv::Point2f>> facialLandmarks;
  void drawPolyline(const std::vector<cv::Point2f>& points,const int start,const int end,bool closed);
  void drawLandmarks(std::vector<cv::Point2f>& landmarks);
public:
  FaceDetector();
  void detectFaces(cv::Mat& inFrame);
  void detectFacialLandmarks();
  std::vector<cv::Rect> getDetectedFaces();
  cv::Mat getProcessedFrame();
};

#endif