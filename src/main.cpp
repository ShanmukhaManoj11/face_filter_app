#include "faceDetector.h"

int main(int argc,char** argv){
	std::shared_ptr<FaceDetector> _faceDetector(new FaceDetector()); //face detector
	cv::VideoCapture videoHandle;
	if(argc==1) videoHandle.open(0);
	else videoHandle.open(argv[1]);
	cv::Mat inFrame;
	while(true){
		videoHandle>>inFrame;
		if(inFrame.empty()) break;
		_faceDetector->detectFaces(inFrame); //function call to detect faces
		_faceDetector->detectFacialLandmarks();
		inFrame=_faceDetector->getProcessedFrame();
		cv::imshow("Face Detection",inFrame);
		int k=cv::waitKey(5);
		if(k==27){
			cv::destroyAllWindows();
			break;
		}
	}
}