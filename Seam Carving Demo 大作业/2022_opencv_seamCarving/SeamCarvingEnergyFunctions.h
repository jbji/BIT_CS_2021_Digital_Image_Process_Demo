//
// Created by Li Chunliang on 2022/2/2.
//

#ifndef INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGENERGYFUNCTIONS_H
#define INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGENERGYFUNCTIONS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class SeamCarvingEnergyFunctions {
public:
    static double Func1(int i, int j, const Mat & src,int seed = 0);
    static double Func2(int i, int j, const Mat & src,int seed = 0);
    static double FuncCanny(int i, int j, const Mat & src, int seed = 0);
    static double FuncSaliency(int i, int j, const Mat & src, int seed = 0);
    //if there's a cache in the function,
    // the seed controls if the function should update the cache.
};



#endif //INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGENERGYFUNCTIONS_H
