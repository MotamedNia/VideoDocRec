//
// Created by hossein on 11/5/18.
//
#include "assessorBase.h"

class VOL: public assessorBase{
public:
    VOL(int windowSize, int step) : assessorBase(windowSize, step) {}

    double qualityAssess(Mat image){
        if (image.type() == CV_8UC3){ //TODO tracing required
            cvtColor(image, image, COLOR_RGB2GRAY);
        }

        Mat laplacian;
        Laplacian(image, laplacian, CV_64F);
        Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(laplacian, mean, stddev, Mat());
        double score = stddev.val[0] * stddev.val[0];


        return score;
    }

};
