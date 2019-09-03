#include "faceDetector.h"

FaceDetector::FaceDetector(): inWidth(300), inHeight(300), inScaleFactor(1.0),
confidenceThreshold(0.7), meanVal(104.0,177.0,123.0), 
configFile("../models/opencv_face_detector.pbtxt"),
weightFile("../models/opencv_face_detector_uint8.pb"){
	net=cv::dnn::readNetFromTensorflow(weightFile,configFile);
	facemark=cv::face::FacemarkLBF::create();
	facemark->loadModel("../models/lbfmodel.yaml");
}

std::vector<cv::Rect> FaceDetector::getDetectedFaces(){
	return this->faces;
}

cv::Mat FaceDetector::getProcessedFrame(){
	return this->frame;
}

void FaceDetector::detectFaces(cv::Mat& inFrame){
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

void FaceDetector::drawPolyline(const std::vector<cv::Point2f>& points,const int start,const int end,bool closed){
	std::vector<cv::Point> _points;
	for(int i=start;i<=end;i++){
		_points.push_back(cv::Point(points[i].x,points[i].y));
	}
	polylines(this->frame,_points,closed,cv::Scalar(255,200,0),2,16);
}

void FaceDetector::drawLandmarks(std::vector<cv::Point2f>& landmarks){
	if(landmarks.size()==68){
		drawPolyline(landmarks,0,16,false); //jaw line
		drawPolyline(landmarks,17,21,false); //left eyebrow
		drawPolyline(landmarks,22,26,false); //right eyebrow
		drawPolyline(landmarks,27,30,false); //nose bridge
		drawPolyline(landmarks,30,35,true); //lower nose
		drawPolyline(landmarks,36,41,true); //left eye
		drawPolyline(landmarks,42,47,true); //right eye
		drawPolyline(landmarks,48,59,true); //outer lip
		drawPolyline(landmarks,60,67,true); //inner lip
	}
	else{
		for(int i=0;i<landmarks.size();i++){
			cv::circle(this->frame,landmarks[i],3,cv::Scalar(255,200,0),cv::FILLED);
		}
	}
}

void FaceDetector::detectFacialLandmarks(){
	facialLandmarks.clear();
	bool success=facemark->fit(this->frame,faces,facialLandmarks);
	if(success){
		for(auto landmarks: facialLandmarks){
			drawLandmarks(landmarks);
		}
	}
}