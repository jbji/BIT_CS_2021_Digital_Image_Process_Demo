//
// Created by Li Chunliang on 2022/1/14.
//

#include "SuperPixelSLIC.h"
#include <stack>

using namespace std;

LABXY::LABXY(): l(0), a(0), b(0), x(-1), y(-1){}
LABXY::LABXY(double l, double a, double b, double x, double y) : l(l), a(a), b(b), x(x), y(y) {}

void LABXY::setLABXY(double _l, double _a, double _b, double _x, double _y) {
    LABXY::l = _l; LABXY::a = _a; LABXY::b = _b; LABXY::x = _x; LABXY::y = _y;
}
void LABXY::print() const {
    printf("LABXY: %lf,%lf,%lf,%lf,%lf\n", l,a,b,x,y);
}

SuperPixelSLIC::SuperPixelSLIC(Mat & src, int S_grid_interval)
        : src(src), S(S_grid_interval), rows(src.rows), cols(src.cols),
          labels(rows, cols, CV_32S, -1), /*Initialize Label & Distance*/
        distance(rows,cols,CV_64F,FLT_MAX){
    //Convert src Matrix to CIELAB
    cvtColor(src, src_lab, COLOR_BGR2Lab);
    //Pre-calculations for grid.
    gridRows = rows / S;
    gridCols = cols / S;
    S = min(src.rows / gridRows, src.cols / gridCols); //make the S larger.
    /*Initialization*/
    K = gridRows * gridCols;
    clusterCenters.resize(K);
    Mat grad;
    calcGradient(src,grad); //Gradient
    for(int i = 0; i < gridRows; i++){
        for(int j = 0 ; j < gridCols; j++){
            //Initialize cluster centers by sampling pixels at regular grid steps S.
            int x = S/2 + i * S; int y = S/2 + j * S;
            //Move Cluster Centers to the lowest gradient position in a 3×3 neighborhood.
            int lowest_grad = grad.at<short>(x, y);
            int lg_x = x; int lg_y = y; //'lg' for lowest gradient
            for(int p = -1; p <= 1; p++){
                for(int q = -1; q <= 1; q++){
                    //如果不超出图像边界
                    if(x+p >=0 && x+p < rows && y+q >=0 && y+q < cols){
                        //获得梯度值并记录最小梯度值
                        int cgrad = grad.at<short>(x+p,y+q);
                        if(cgrad < lowest_grad){
                            lowest_grad = cgrad; lg_x = x + p; lg_y = y + q;
                        }
                    }

                }
            }
            Vec<uchar, 3> tmp = src_lab.at<Vec3b>(lg_x, lg_y);
            clusterCenters[i * gridCols + j]
                .setLABXY(tmp[0], tmp[1], tmp[2], lg_x, lg_y);
        }
    }
}

void SuperPixelSLIC::iterate(int times, double compactness){
    double factor = 1.0 / (( S * compactness ) * ( S * compactness )); //pre calculate the factor
    for(;times > 0; times--){
        /* Assignment */
        for(int i = 0 ; i < clusterCenters.size(); i++){
            auto & tmp = clusterCenters[i];
            if(tmp.x == -1 || tmp.y == -1) continue;
            //search around the cluster center
            for(int p = max(0, (int)tmp.x - S); p <= min(rows - 1, (int)tmp.x + S); p++){
                for(int q = max(0, (int)tmp.y - S); q <= min(cols - 1, (int)tmp.y + S); q++){
                    //update distance for a pixel
                    Vec<uchar, 3> lab = src.at<Vec3b>(p, q);
                    LABXY li = tmp, lj = LABXY(lab[0], lab[1], lab[2], p, q);
                    double dc2 = (li.l - lj.l)*(li.l - lj.l) + (li.a - lj.a)*(li.a - lj.a)
                            + (li.b - lj.b)*(li.b - lj.b);
                    double ds2 = (li.x - lj.x)*(li.x - lj.x) + (li.y - lj.y)*(li.y - lj.y);
                    double d = dc2 + ds2 * factor;
                    if(d <= distance.at<double>(p,q)){
                        distance.at<double>(p,q) = d;
                        labels.at<int>(p, q) = i;
                    }
                }
            }
        }
        /*Update*/
        for(int i = 0 ; i < clusterCenters.size(); i++){
            auto & tmp = clusterCenters[i];
            if(tmp.x == -1 || tmp.y == -1) continue;
            //pixels belonging to the cluster center are counted
            double s[5] = {0}, cnt = 0; // s for sum, cnt for count
            for(int p = max(0, (int)tmp.x - S); p <= min(rows - 1, (int)tmp.x + S); p++){
                for(int q = max(0, (int)tmp.y - S); q <= min(cols - 1, (int)tmp.y + S); q++){
                    if(labels.at<int>(p, q) == i){
                        Vec<uchar,3> tmpVec = src_lab.at<Vec3b>(p, q); //GetLAB of Pixel
                        s[0] += tmpVec[0]; s[1] += tmpVec[1]; s[2] += tmpVec[2]; //LAB
                        s[3] += p; s[4] += q; // XY
                        cnt++;
                    }
                }
            }
            /*recalculate the cluster center*/
            if(cnt !=0) {
                tmp.setLABXY(s[0] / cnt, s[1] / cnt, s[2] / cnt,s[3] / cnt, s[4] / cnt);
            }else{
                tmp.setLABXY(0,0,0,-1,-1);
                K--;
            }
        }
    }
}

Mat SuperPixelSLIC::getLabels(){
    return labels;
}

void SuperPixelSLIC::calcGradient(Mat & src, Mat & grad){
    Mat src_gray; //假定输入是彩色图像
    cvtColor(src, src_gray, COLOR_RGB2GRAY);
    // 创建 grad_x 和 grad_y 矩阵
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    // 求 X方向梯度
    Sobel( src_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    // 求Y方向梯度
    Sobel( src_gray, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    // 合并梯度(近似)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
}

void SuperPixelSLIC::enforceConnectivity(int minSegSize) {
    if (minSegSize == 0 || minSegSize > 100) return;
    //四邻域连接
    const int dx4[4] = { -1,  0,  1,  0 };
    const int dy4[4] = {  0, -1,  0,  1 };
    const int magic = max(3, cols * rows / K / int(100.0/(float)minSegSize + 0.5));
    Mat l_src = labels;
    labels = Mat(rows, cols, CV_32S, -1);
    int label = 0;
    int adjLabel = 0; //adjacent label
    vector<int> componentPoints(rows*cols);
    for (int x = 0; x < rows; x++){
        for (int y = 0; y < cols; y++){
            if (labels.at<int>(x, y) == -1 ){// Seed found
                labels.at<int>(x,y) = label;
                componentPoints[0] = y + x * cols;
                //Find an adjacent label
                for(int i = 0; i < 4; i++){
                    int p = componentPoints[0] / cols + dx4[i];
                    int q = componentPoints[0] % cols + dy4[i];
                    if( p >=0 && p < rows && q >=0 && q < cols){
                        if(labels.at<int>(p,q) != -1){
                            adjLabel = labels.at<int>(p, q);
                            break;
                        }
                    }
                }
                //iterate and grow
                int count = 1;
                for( int t = 0; t < count; t++){
                    for(int i = 0; i < 4; i++){
                        int p = componentPoints[t] / cols + dx4[i];
                        int q = componentPoints[t] % cols + dy4[i];
                        if( p >=0 && p < rows && q >=0 && q < cols){
                            if(labels.at<int>(p,q) == -1 &&
                                    l_src.at<int>(p,q) == l_src.at<int>(x,y)){
                                componentPoints[count] = q + p * cols;
                                labels.at<int>(p,q) = label;
                                count++;
                            }
                        }
                    }
                }
                //if segment size is no bigger than a threshold,
                // allocate it to previous adjacent segmant
                if(count <= magic){
                    for(int t = 0; t < count; t++){
                        labels.at<int>(componentPoints[t] / cols,
                                       componentPoints[t] % cols) = adjLabel;
                    }
                    label--;
                }
                label++;
            }
        }
    }

//    return labels;
}
