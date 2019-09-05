#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <opencv2/viz/types.hpp>
#include <opencv2/viz/widgets.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>

int main(int argc,char** argv){
	cv::String obj("../models/PLY_Pasha.ply");
	cv::viz::Mesh mesh=cv::viz::readMesh(obj);
	cv::viz::Viz3d window3d("3d model");
	window3d.spin();
}