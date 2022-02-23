//
// Created by Li Chunliang on 2022/2/2.
//

#ifndef INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGRESIZE_H
#define INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGRESIZE_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "SeamCarvingEnergyFunctions.h"


using namespace cv;
using namespace std;

typedef pair<int,Vec3b> ColPixel;
typedef vector<ColPixel> Seam;
typedef vector<Seam> Seams;

// vector< vector< pair<int cols,Vec3b pixel> >  is a seam > is a series of seams

enum ResizeMode{
    MIXED, REMOVAL, INSERTION
};
class SeamCarvingResize {

protected:
    int cols;
    int rows;
    bool isTransposed; //transpose mark
    Mat src, energy_map;
    double (*energyFunction) (int ,int , const Mat &, int);
    int seed;

public:
    string winname;
    explicit SeamCarvingResize(Mat & src, double (*_energyFunction) (int ,int , const Mat &, int) = SeamCarvingEnergyFunctions::Func1);
    /*Aspect Ratio Change*/
    void changeAspectRatio(double col_ratio, double row_ratio, ResizeMode mode = REMOVAL);
    /*Fundamental Functions*/
    virtual Seams removeCols(int num, bool mute = false);
    void removeRows(int num);
    void insertCols(int num, int isSubInsert = false);
    void insertRows(int num);
    void show();
protected:
    void calcEnergyMap();
    /*could be overrided for improvement*/
    virtual void calcCumulativeMinimumEnergy(Mat &cmin_energy);
};


#endif //INC_2022_OPENCV_SEAMCARVING_SEAMCARVINGRESIZE_H
