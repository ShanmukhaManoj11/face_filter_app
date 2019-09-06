#include <math.h>
#include "faceDetector.h"

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>

int main(int argc,char** argv){
	double camZ=1500.0;
	cv::Vec3f camPos(0,0,(float)camZ);
	cv::Vec3f camFocalPoint(0,0,0);
	cv::Vec3f camYdir(0,0,0);
	cv::Affine3f viewPose=cv::viz::makeCameraPose(camPos,camFocalPoint,camYdir);

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
	window3d.setViewerPose(viewPose);

	std::shared_ptr<FaceDetector> _faceDetector(new FaceDetector());

	while(!window3d.wasStopped()){
		videoHandle>>frame;
		if(frame.empty()) break;

		_faceDetector->detectFaces(frame);
		_faceDetector->detectFacialLandmarks();
		frame=_faceDetector->getProcessedFrame();
		wframe.setImage(frame);
		wframe.setPose(framePose);

		std::vector<cv::Rect> _faces=_faceDetector->getDetectedFaces();
		if(!_faces.empty()){
			double cx=_faces[0].x;
			double cy=_faces[0].y;
			/*
			cv::Point3d origin;
			cv::Vec3d direction;
			window3d.converTo3DRay(cv::Point3d(cx,cy,0),origin,direction);
			//std::cout<<direction[0]<<" "<<direction[1]<<" "<<direction[2]<<std::endl;
			if(direction[2]!=0){
				double worldx=(direction[0]/direction[2])*camZ;
				double worldy=(direction[1]/direction[2])*camZ;
				cv::Affine3f meshPose(cv::Vec3f(0,0,0),cv::Vec3f(worldx,worldy,0));
				objmesh.setPose(meshPose);
			}
			*/
			cv::Affine3f meshPose(cv::Vec3f(0,0,0),cv::Vec3f(cx,cy,50));
			objmesh.setPose(meshPose);
		}

		window3d.spinOnce(1,true);
	}
}