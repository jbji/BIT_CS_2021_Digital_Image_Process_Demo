#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "SuperPixelSLIC.h"

using namespace std;
using namespace cv;
using namespace ximgproc;

int region_size = 100;
int num_iterations = 10;
char img_str[] = "Sunday Afternoon.jpg";

void showLabels(Mat label, const char * winname);
void OurImplementation(Mat & src);
void OpenCVImplementation(Mat & src);

int main() {
    Mat src = imread(img_str,IMREAD_COLOR), dst;
    OurImplementation(src);
//    OpenCVImplementation(src);
    waitKey(0);
    return 0;
}

void showLabels(Mat label, const char * winname){
    //为标签添加伪彩色以便观察
    Mat dst(label.rows, label.cols, CV_8UC3);
    for(int i = 0; i < label.rows; i++){
        auto data = dst.ptr<Vec3b>(i);
        for(int j = 0; j < label.cols; j++){
            int c = label.at<int>(i,j);
            data[j] = cv::Vec3b(5+c%50*5, 5+(c/25)%25*10,
                                5+( (c+127)/25)%25*10);
        }
    }
    cvtColor( dst, label , COLOR_Lab2BGR);//转BGR
    applyColorMap(label,dst,COLORMAP_JET);//热力图方式
    imshow(winname, dst);
}


void OurImplementation(Mat & src){
    SuperPixelSLIC s(src,region_size);
    s.iterate(num_iterations);
    s.enforceConnectivity();
    showLabels(s.getLabels(), "labels - OurImplementation");
}

void OpenCVImplementation(Mat & src){
    Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(
            src,SLIC, region_size);
    slic->iterate(num_iterations);
    slic->enforceLabelConnectivity();
    /*labels*/
    Mat labels;
    slic->getLabels(labels);//获取labels
    showLabels(labels, "labels - OpenCVImplementation");
}