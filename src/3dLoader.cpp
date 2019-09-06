#include <math.h>
#include "faceDetector.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>

int main(int argc,char** argv){
	cv::String obj("../models/PLY_Pasha.ply");
	cv::viz::WMesh objmesh( cv::viz::Mesh::load(obj) ); //3d object mesh as widget on viz3d window

	cv::VideoCapture videoHandle;
	if(argc==1) videoHandle.open(0);
	else videoHandle.open(argv[1]);
	int frameWidth=videoHandle.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeight=videoHandle.get(cv::CAP_PROP_FRAME_HEIGHT);
	cv::Mat frame;
	cv::Affine3f framePose(cv::Vec3f(M_PI,0,0),cv::Vec3f(0,0,0)); //set affine transformation to rotate frame on widget
	videoHandle>>frame;
	cv::viz::WImage3D wframe(frame,cv::Size2d(frameWidth,frameHeight)); //camera frame as widget on viz3d window

	cv::viz::Viz3d window3d("3d model");
	window3d.showWidget("object",objmesh);
	window3d.showWidget("frame",wframe);
	window3d.setFullScreen(true);

	std::shared_ptr<FaceDetector> _faceDetector(new FaceDetector());

	while(!window3d.wasStopped()){
		videoHandle>>frame;
		if(frame.empty()) break;
		_faceDetector->detectFaces(frame);
		_faceDetector->detectFacialLandmarks();
		frame=_faceDetector->getProcessedFrame();
		
		wframe.setImage(frame);
		wframe.setPose(framePose);
		window3d.spinOnce(1,true);
	}
}