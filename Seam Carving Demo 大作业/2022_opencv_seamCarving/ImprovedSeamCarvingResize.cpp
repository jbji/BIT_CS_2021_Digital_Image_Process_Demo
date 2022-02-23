//
// Created by Li Chunliang on 2022/2/4.
//

#include "ImprovedSeamCarvingResize.h"

using namespace std;

void ImprovedSeamCarvingResize::calcCumulativeMinimumEnergy(Mat &cmin_energy) {
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    char a, b, c;
    int C_L, C_U, C_R;
    double M_a, M_b, M_c;
    //forward energy
    for(int i = 1; i < rows; i++){
        for(int j = 0; j < cols; j++){
            a = src_gray.at<char>(i,j==0?cols-1:j-1);
            b = src_gray.at<char>(i-1,j);
            c = src_gray.at<char>(i,j==cols-1?0:j+1);
            C_U = abs(c-a);
            C_L = C_U + abs(b-a);
            C_R = C_U + abs(b-c);
            if(j != 0) M_a = energy_map.at<double>(i - 1, j - 1) + C_L;
            M_b = energy_map.at<double>(i - 1, j) + C_U;
            if(j != cols-1) M_c = energy_map.at<double>(i - 1, j + 1) + C_R;
            if( j == 0 ){
                cmin_energy.at<double>(i,j) = energy_map.at<double>(i, j) + min(M_c, M_b );
            }else if(j == cols - 1){
                cmin_energy.at<double>(i,j) = energy_map.at<double>(i, j) + min(M_a, M_b );
            }else{
                cmin_energy.at<double>(i,j) = energy_map.at<double>(i, j) + min( min(M_a, M_b ), min(M_c, M_b ) );
            }
        }
    }
}

Seams ImprovedSeamCarvingResize::removeCols(int num, bool mute) { //returns Seams being removed, relative to the original image
    Seams result; //for recording seams
    begin:
    if(num <= 0) {
        /* for recording, seam cordinate post-process*/
        //update seams_index
        for(int i = (int)result.size() - 2; i >= 0; i--){ //被参考的seam
            //from (int)result.size() - 1 to i, calc diff and add
            for(int j = (int)result.size() - 1; j > i; j--){ //要修正的seam
                Seam & cur = result[j];
                Seam & prev = result[i];
                for(int idx = 0; idx < rows; idx++){
                    if(cur[idx].first >= prev[idx].first){
                        cur[idx].first++;
                    }
                }
            }
        }
        return result;
    }
    if(num > cols) num = cols;
    /* Step 1. try to calculate(and show) the energy map */
    calcEnergyMap();

    /* Step 2. calc cumulative minimul energy, which indicates vertical seam. */
    Mat cmin_energy = energy_map; // calcCumulativeMinimumEnergy
    calcCumulativeMinimumEnergy(cmin_energy);

    /* Step 3. find seam */
    int seam_i = -1;
    double seam_min = FLT_MAX;
    for(int j = 0 ; j < cols; j++){
        if(cmin_energy.at<double>(rows-1, j) < seam_min){
            seam_min = cmin_energy.at<double>(rows - 1, j);
            seam_i = j;
        }
    }

    /* Step 4. Remove Vertical Seam*/
    /*Improve special*/
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    char a, b, c;
    int C_L, C_U, C_R;
    double M_a, M_b, M_c;

    Seam current_seam; //for recording
    Mat tmp = Mat(rows, cols-1, CV_8UC3); // this stores the tmp result
    for(int i = rows-1; i >= 0; i-- ){
        //jump over seam
        int k = 0;
        for(int j = 0; j < cols; j++){
            if(j == seam_i) {
                /* for recording*/
                ColPixel pp = make_pair(seam_i, src.at<Vec3b>(i,j));
                current_seam.push_back(pp);
                continue;
            }
            tmp.at<Vec3b>(i,k++) = src.at<Vec3b>(i,j);
        }
        //update seam index when it's not the first row, improved
        if(i!= 0){
            seam_min = FLT_MAX;
            a = src_gray.at<char>(i,seam_i==0?cols-1:seam_i-1);
            b = src_gray.at<char>(i-1,seam_i);
            c = src_gray.at<char>(i,seam_i==cols-1?0:seam_i+1);
            C_U = abs(c-a);
            C_L = C_U + abs(b-a);
            C_R = C_U + abs(b-c);
            if(seam_i != 0) M_a = energy_map.at<double>(i - 1, seam_i - 1) + C_L;
            M_b = energy_map.at<double>(i - 1, seam_i) + C_U;
            if(seam_i != cols-1) M_c = energy_map.at<double>(i - 1, seam_i + 1) + C_R;
            int j = seam_i;
            if( seam_i == 0 ){
                if(M_b < seam_min){
                    seam_min = M_b;
                    j = seam_i;
                }
                if(M_c < seam_min){
                    seam_min = M_b;
                    j = seam_i+1;
                }
            }else if(seam_i == cols - 1){
                if(M_a < seam_min){
                    seam_min = M_a;
                    j = seam_i -1;
                }
                if(M_b < seam_min){
                    seam_min = M_b;
                    j = seam_i;
                }
            }else{
                if(M_a < seam_min){
                    seam_min = M_a;
                    j = seam_i -1;
                }
                if(M_b < seam_min){
                    seam_min = M_b;
                    j = seam_i;
                }
                if(M_c < seam_min){
                    seam_min = M_b;
                    j = seam_i+1;
                }
            }
            seam_i = j;
        }
    }
    src = tmp;
    rows = tmp.rows;
    cols = tmp.cols;
    result.push_back(current_seam); // for recording

    /* Step 5. Loop */
    if(!mute){
        show();
        waitKey(1);
    }
    num--;
    goto begin;
}

ImprovedSeamCarvingResize::ImprovedSeamCarvingResize(Mat mat, double (*param)(int, int, const Mat &, int)) :
    SeamCarvingResize(mat,param){
    winname = "Improved Seam Carving Result";
}