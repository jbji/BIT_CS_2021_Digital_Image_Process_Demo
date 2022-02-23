//
// Created by Li Chunliang on 2022/2/2.
//

#include "SeamCarvingEnergyFunctions.h"

#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>

using namespace cv;
using namespace std;
using namespace saliency;

int Func1_sub(int i, int j, const Mat & src, int channel) {
    //calc grad on x/y direction, with one channel
    int x_part = 0, y_part = 0;
    x_part = src.at<Vec3b>((i != src.rows - 1 ? i+1 : 0),j)[channel] - src.at<Vec3b>(i,j)[channel];
    y_part = src.at<Vec3b>(i,(j != src.cols - 1 ? j+1 : 0))[channel] - src.at<Vec3b>(i,j)[channel];
    return abs(x_part) + abs(y_part);
}

double SeamCarvingEnergyFunctions::Func1(int i, int j, const Mat &src,int seed) {

    // if i or j is invalid, return -1;
    if(i < 0 || i >= src.rows || j <0 || j >= src.cols){ return -1;}

    int b, g, r;
    b = Func1_sub(i, j, src, 0);
    g = Func1_sub(i, j, src, 1);
    r = Func1_sub(i, j, src, 2);

    return b + g + r;
}

double SeamCarvingEnergyFunctions::FuncCanny(int i, int j, const Mat &src, int seed) {
    //结合Canny边缘检测的能量函数计算方法
    static int seed_cache = -1;
    static Mat src_gray, detected_edges;
    static std::vector<Mat> channels;
    static double detected_max;
    if(seed_cache != seed){
        seed_cache = seed;
        cvtColor(src,detected_edges,COLOR_BGR2GRAY);
        int kernel_size = 5, lowThreshold = 20, highThreshold = 100;
//        blur( src_gray, detected_edges, Size(5,5) );
        Canny( detected_edges, detected_edges, lowThreshold, highThreshold, kernel_size );
        detected_max = *max_element(detected_edges.begin<char>(), detected_edges.end<char>());
    }
//    return detected_edges.at<char>(i,j);
//    return detected_edges.at<char>(i,j) * Func1(i,j,src);
    return (detected_edges.at<char>(i,j)/detected_max+1) * Func1(i,j,src);
}


double SeamCarvingEnergyFunctions::FuncSaliency(int i, int j, const Mat &src, int seed) {
    //结合Saliency Map的方法
    static int seed_cache = -1;
    static Mat saliency_src;
    static std::vector<Mat> channels;
    static double detected_max;
    if(seed_cache != seed){
        seed_cache = seed;
        saliency_src = src.clone();
        StaticSaliencySpectralResidual s;
        s.computeSaliency(saliency_src, saliency_src);
        saliency_src *= 255;
        detected_max = *max_element(saliency_src.begin<float>(), saliency_src.end<float>());
    }
//    return saliency_src.at<float>(i,j);
    return saliency_src.at<float>(i,j) * Func1(i,j,src);
//    return (saliency_src.at<float>(i,j)/detected_max +0.3 ) * Func1(i,j,src);
}

int Func2_sub(int i, int j, const Mat & src, int channel) {
    //水平垂直差分法+罗伯特差分法
    //calc grad on x/y direction, with one channel
    int x_part = 0, y_part = 0, a_part = 0, b_part = 0;
    x_part = src.at<Vec3b>((i != src.rows - 1 ? i+1 : 0),j)[channel] - src.at<Vec3b>(i,j)[channel];
    y_part = src.at<Vec3b>(i,(j != src.cols - 1 ? j+1 : 0))[channel] - src.at<Vec3b>(i,j)[channel];
    a_part = src.at<Vec3b>((i != src.rows - 1 ? i+1 : 0),j)[channel] - src.at<Vec3b>(i,(j != src.cols - 1 ? j+1 : 0))[channel];
    b_part = src.at<Vec3b>((i != src.rows - 1 ? i+1 : 0),(j != src.cols - 1 ? j+1 : 0))[channel] - src.at<Vec3b>(i,j)[channel];

//    return max( max(abs(x_part) , abs(y_part)) , max(abs(a_part) , abs(b_part)) );
    return abs(x_part) + abs(y_part) + abs(a_part) + abs(b_part);
}

double SeamCarvingEnergyFunctions::Func2(int i, int j, const Mat &src, int seed) {
    // if i or j is invalid, return -1;
    if(i < 0 || i >= src.rows || j <0 || j >= src.cols){ return -1;}

    int b, g, r;
    b = Func2_sub(i, j, src, 0);
    g = Func2_sub(i, j, src, 1);
    r = Func2_sub(i, j, src, 2);

    return b + g + r;
}


