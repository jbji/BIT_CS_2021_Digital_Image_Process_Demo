//
// Created by Li Chunliang on 2022/1/14.
//

#ifndef INC_2022_OPENCV_SLIC_SUPERPIXELSLIC_H
#define INC_2022_OPENCV_SLIC_SUPERPIXELSLIC_H

#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;

struct LABXY{
    double l,a,b,x,y;
    LABXY();
    LABXY(double l, double a, double b, double x, double y);
    void setLABXY(double _l, double _a, double _b, double _x, double _y);
    void print() const;
};

class SuperPixelSLIC{

private:
    //private params
    int rows, cols;
    int gridRows, gridCols;
    int S; // clusterCenters interval
    int K;
    Mat src_lab, &src;
    Mat labels,distance;
    vector<LABXY> clusterCenters; //聚类中心

public:
    /*Initialization Step For SLIC*/
    explicit SuperPixelSLIC(Mat & src, int S_grid_interval = 10);
    void iterate(int times = 10, double compactness = 10.0);
    Mat getLabels();
    void enforceConnectivity(int minSegSize = 25);
private:
    //计算梯度
    static void calcGradient(Mat & src, Mat & grad);
};

#endif //INC_2022_OPENCV_SLIC_SUPERPIXELSLIC_H
