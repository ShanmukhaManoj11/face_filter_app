#include "faceDetector.h"

FaceDetector::FaceDetector(): inWidth(300), inHeight(300), inScaleFactor(1.0),
confidenceThreshold(0.7), meanVal(104.0,177.0,123.0), 
configFile("../models/opencv_face_detector.pbtxt"),
weightFile("../models/opencv_face_detector_uint8.pb"){
	net=cv::dnn::readNetFromTensorflow(weightFile,configFile);
}

std::vector<cv::Rect> FaceDetector::getDetectedFaces(){
	return this->faces;
}

cv::Mat FaceDetector::getProcessedFrame(){
	return this->frame;
}

void FaceDetector::detectFace(cv::Mat& inFrame){
	this->frame=inFrame.clone();
	int frameHeight=inFrame.rows;
    int frameWidth=inFrame.cols;
    cv::Mat inputBlob=cv::dnn::blobFromImage(inFrame,inScaleFactor,cv::Size(inWidth,inHeight),
    	meanVal,true,false);
    net.setInput(inputBlob,"data");
    cv::Mat detection=net.forward("detection_out");
    cv::Mat detectionMat(detection.size[2],detection.size[3],CV_32F,detection.ptr<float>());
    if(!faces.empty()) faces.clear();
    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence=detectionMat.at<float>(i, 2);
        if(confidence>confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);
            faces.push_back(cv::Rect(cv::Point(x1,y1),cv::Point(x2,y2)));
            cv::rectangle(this->frame,cv::Point(x1,y1),cv::Point(x2,y2),
            	cv::Scalar(0,255,0),2,4);
        }
    }
}