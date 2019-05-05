//
// Created by hossein on 11/6/18.
//
#include "assessorBase.h"

class rank_bluro: public assessorBase{
public:
    rank_bluro(int windowSize, int step) : assessorBase(windowSize, step) {}

    double qualityAssess(Mat image){
        Mat temp = image;
        image.convertTo(image, CV_32F, 1 / 255.0);
        Mat gx, gy;
        Sobel(image, gx, CV_32F, 1, 0, 1);
        Sobel(image, gy, CV_32F, 0, 1, 1);
        Mat mag, angle;
        cartToPolar(gx, gy, mag, angle, 1);
        double s = cv::sum(mag)[0];
        double score = s / (mag.rows * mag.cols);

        return score;
    }

};