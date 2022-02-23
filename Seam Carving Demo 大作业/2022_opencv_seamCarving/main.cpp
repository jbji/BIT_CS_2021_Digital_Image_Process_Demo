#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "SeamCarvingResize.h"
#include "ImprovedSeamCarvingResize.h"

using namespace cv;
using namespace std;

void experiment0(const string& img){
    Mat src = imread(img,IMREAD_COLOR), src_1, src_2, src_3;
    src_1 = src.clone(); src_2 = src.clone(); src_3 = src.clone();
    SeamCarvingResize s1(src_1, SeamCarvingEnergyFunctions::Func1);
    s1.winname = "Original Removal";
    s1.changeAspectRatio(3,4,REMOVAL);
    ImprovedSeamCarvingResize s2(src_2, SeamCarvingEnergyFunctions::FuncCanny);
    s2.winname = "Improved Removal With Canny";
    s2.changeAspectRatio(3,4,REMOVAL);
    ImprovedSeamCarvingResize s3(src_3, SeamCarvingEnergyFunctions::FuncSaliency);
    s3.winname = " Improved Removal With Saliency";
    s3.changeAspectRatio(3,4,REMOVAL);
}
void experiment1(const string& img, const string& num){
    Mat src = imread(img,IMREAD_COLOR), src_1, src_2, src_3;
    src_1 = src.clone(); src_2 = src.clone(); src_3 = src.clone();
    SeamCarvingResize s1(src_1, SeamCarvingEnergyFunctions::Func1);
    s1.winname = num + " Removal";
    s1.changeAspectRatio(3,4,REMOVAL);
    SeamCarvingResize s2(src_2, SeamCarvingEnergyFunctions::Func1);
    s2.winname = num + " Insertion";
    s2.changeAspectRatio(3,4,INSERTION);
    SeamCarvingResize s3(src_3, SeamCarvingEnergyFunctions::Func1);
    s3.winname = num + " Mixed";
    s3.changeAspectRatio(3,4,MIXED);
}

void experiment2(const string& img, const string& num){
    Mat src = imread(img,IMREAD_COLOR), src_1, src_2;
    src_1 = src.clone(); src_2 = src.clone();
    SeamCarvingResize s1(src_1, SeamCarvingEnergyFunctions::Func1);
    s1.winname = num + " Original";
    s1.changeAspectRatio(3,4,REMOVAL);
    ImprovedSeamCarvingResize s2(src_2, SeamCarvingEnergyFunctions::Func1);
    s2.winname = num + " Improved";
    s2.changeAspectRatio(3,4,REMOVAL);
}

void experiment3(const string& img, const string& num){
    Mat src = imread(img,IMREAD_COLOR), src_1, src_2;
    src_1 = src.clone(); src_2 = src.clone();

    ImprovedSeamCarvingResize s1(src_1, SeamCarvingEnergyFunctions::Func1);
    s1.winname = num + " E1";
    s1.changeAspectRatio(3,4,REMOVAL);

    ImprovedSeamCarvingResize s2(src_2, SeamCarvingEnergyFunctions::Func2);
    s2.winname = num + " E2";
    s2.changeAspectRatio(3,4,REMOVAL);

    ImprovedSeamCarvingResize s3(src_2, SeamCarvingEnergyFunctions::FuncCanny);
    s3.winname = num + " Canny";
    s3.changeAspectRatio(3,4,REMOVAL);

    ImprovedSeamCarvingResize s4(src_2, SeamCarvingEnergyFunctions::FuncSaliency);
    s4.winname = num + " Saliency";
    s4.changeAspectRatio(3,4,REMOVAL);
}


int main() {

//    experiment0("bit.png"); //for generating cover
//    /*Experiment 1*/
//    experiment1("bliss.jpg","4.1-1"); //bliss
    experiment1("1-1.png","4.1-2");  //sea
//
//    /*Experiment 2*/
//    experiment2("img4mini.png","4.2-1"); //geometric
    experiment2("rings.png","4.2-2"); //rings
//    experiment2("wavesinsea.png","4.2-3"); //wavesinsea

    /*Experiment 3*/
    experiment3("4.png","4.3-1"); //pot
//    experiment3("3.png","4.3-2"); //building
//    experiment3("imgmini.png","4.3-3"); //img
//    experiment3("lx.png","4.3-4"); //lx

    waitKey(0);
    return 0;
}
