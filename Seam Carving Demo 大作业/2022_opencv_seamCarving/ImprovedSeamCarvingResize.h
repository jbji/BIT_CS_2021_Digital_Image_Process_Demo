//
// Created by Li Chunliang on 2022/2/4.
//

#ifndef INC_2022_OPENCV_SEAMCARVING_IMPROVEDSEAMCARVINGRESIZE_H
#define INC_2022_OPENCV_SEAMCARVING_IMPROVEDSEAMCARVINGRESIZE_H

#include "SeamCarvingResize.h"

class ImprovedSeamCarvingResize : public SeamCarvingResize{
protected:
    /* Improved Seam Detection*/
    void calcCumulativeMinimumEnergy(Mat &cmin_energy) override;
public:
    ImprovedSeamCarvingResize(Mat mat, double (*param)(int, int, const Mat &, int) = SeamCarvingEnergyFunctions::Func1);
    Seams removeCols(int num, bool mute) override;
};


#endif //INC_2022_OPENCV_SEAMCARVING_IMPROVEDSEAMCARVINGRESIZE_H
