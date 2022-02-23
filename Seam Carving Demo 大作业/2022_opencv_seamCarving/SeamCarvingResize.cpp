//
// Created by Li Chunliang on 2022/2/2.
//

#include "SeamCarvingResize.h"
#include <cmath>

SeamCarvingResize::SeamCarvingResize(
        Mat &src, double (*_energyFunction)(int, int, const Mat &, int)
        ): src(src), rows(src.rows), cols(src.cols), energyFunction(_energyFunction),
           energy_map(rows, cols, CV_64F, FLT_MAX), isTransposed(false), seed(0){
    if(winname.empty()) winname = "Seam Carving Result";
}


Seams SeamCarvingResize::removeCols(int num, bool mute) { //returns Seams being removed, relative to the original image
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
        //update seam index when it's not the first row
        if(i!= 0){
            seam_min = FLT_MAX;
            for(int p = max(seam_i - 1, 0) ; p <= min(seam_i +1, cols-1); p++){
                if(cmin_energy.at<double>(i-1, p) < seam_min){
                    seam_min = cmin_energy.at<double>(i - 1, p);
                    seam_i = p;
                }
            }
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

void SeamCarvingResize::calcCumulativeMinimumEnergy(Mat &cmin_energy) {
    double M_a, M_b, M_c;
    for(int i = 1; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(j != 0) M_a = energy_map.at<double>(i - 1, j - 1);
            M_b = energy_map.at<double>(i - 1, j);
            if(j != cols-1) M_c = energy_map.at<double>(i - 1, j + 1);
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

void SeamCarvingResize::calcEnergyMap() {
//    Mat tmp(rows, cols, CV_8UC1); //this stores the energy map to show
    energy_map = Mat(rows, cols, CV_64F, FLT_MAX); // energy map
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
//            tmp.at<char>(i,j) = (char) ((int)(*energyFunction)(i, j, src, seed) % 255);
            energy_map.at<double>(i, j) = (int)(*energyFunction)(i, j, src, seed);
        }
    }
//    imshow("Energy Map", tmp);
//    moveWindow("Energy Map", 300,700);
//    waitKey(1);
    seed++;
}

void SeamCarvingResize::show() {
    if(!isTransposed){
        imshow(winname, src);
    }else{
        transpose(src, src);
        imshow(winname, src);
        transpose(src, src);
    }

//    moveWindow("Seam Carving Result", 300,300);
}


void SeamCarvingResize::removeRows(int num) {

    transpose(src, src);
    swap(cols, rows);

    isTransposed = true;
    removeCols(num);
    isTransposed = false;

    transpose(src, src);
    swap(cols, rows);
}

void SeamCarvingResize::insertCols(int num, int isSubInsert) {
    if(num <= 0) return;
    /* Break insertion into pieces*/
    if(!isSubInsert){
        int threshold = cols/5;
        if(num > threshold){
            int insert_times = (num+threshold-1)/threshold;
            for(int i = 0; i < insert_times-1; i++){
                cout << "Insertion " << i+1 << "/" << (num % threshold == 0? insert_times-1: insert_times) << ":" << endl;
                insertCols(threshold, true);
            }
            if(num % threshold) cout << "Insertion " << insert_times << "/" << insert_times << ":" << endl;
            insertCols(num % threshold, true);
            return;
        }
    }
    /* Step 1. find seams to duplicate */
    Mat src_bak = src.clone();
    cout << "Calculating Seams for insertion, please wait..." << endl;
    Seams s = removeCols(num, true);
    cout << "Inserting..." << endl;
    src = src_bak;
    rows = src.rows; cols = src.cols;

    /* Step 2. Duplicate Seams */
    for(int seam_absi = 0; seam_absi < num; seam_absi++){
        int seam_i = seam_absi % (int)s.size();
        Seam seam = s[seam_i];
        Mat tmp = Mat(rows, cols+1, CV_8UC3);
        for(int i = rows - 1; i >= 0; i--){
            /* copy seam */
            ColPixel pixel = seam[rows-1-i];
            int k = 0;
            for(int j = 0; j < cols; j++){
                tmp.at<Vec3b>(i, k++) = src.at<Vec3b>(i, j);
                if(j == pixel.first){
                    //average pixel of the seam; src.at<Vec3b>(i, j-1); src.at<Vec3b>(i, j+1);
                    float cnt = 1;
                    Vec3f tmpVec = pixel.second;
                    if(j-1>=0){
                        tmpVec += src.at<Vec3b>(i, j-1);
                        cnt++;
                    }
                    if(j+1 <= cols-1) {
                        tmpVec += src.at<Vec3b>(i, j+1);
                        cnt++;
                    }
                    tmpVec /= cnt;
                    tmp.at<Vec3b>(i, k++) = pixel.second;
                    continue;
                }
            }
        }
        src = tmp;
        cols++;

        //update seams_index
        for(Seam & ss : s){
            for(int i = 0; i < rows; i++){
                if(ss[i].first >= seam[i].first){
                    ss[i].first++;
                }
            }
        }
        //display the result
        show();
        waitKey(1);
    }
    cout << "Insertion complete." << endl;
}

void SeamCarvingResize::insertRows(int num) {
    transpose(src, src);
    swap(cols, rows);

    isTransposed = true;
    insertCols(num);
    isTransposed = false;

    transpose(src, src);
    swap(cols, rows);
}

void SeamCarvingResize::changeAspectRatio(double col_ratio, double row_ratio, ResizeMode mode) {
    //e.g. 16:9, 18:9, 21:9, 1:1, 4:3, 3:4
    if(mode == MIXED){
        //MIXED keeps the area of the image the same. 先做insert，再做remove
        double k = row_ratio/col_ratio;
        double s = rows * cols;
        int _rows = (int)sqrt(k*s), _cols = (int)sqrt(s/k);
        //必有一个是insert，另一个是remove。
        if(_cols - cols > 0){
            insertCols(_cols - cols);
            removeRows(rows - _rows);
        }
        if(_rows - rows > 0){
            insertRows(_rows - rows);
            removeCols(cols - _cols);
        }
    }else if(mode == REMOVAL){
//        if((double)cols/(double)rows > col_ratio/row_ratio){
        if(cols*row_ratio > col_ratio*rows){
            //在列相同的情形下，有更少的行，则裁剪列
            removeCols((int)(cols-rows*(col_ratio/row_ratio)));
        }else{
            //在列相同的情形下，有更多的行，则裁剪行
            removeRows((int)(rows-cols*(row_ratio/col_ratio)));
        }
    }else if(mode == INSERTION){
        if(cols*row_ratio > col_ratio*rows){
            //在列相同的情形下，有更少的行，则插入行
            insertRows((int)(cols*row_ratio/col_ratio)-rows);
        }else{
            //在列相同的情形下，有更多的行，则插入列
            insertCols((int)(rows*col_ratio/row_ratio)-cols);
        }
    }
}
