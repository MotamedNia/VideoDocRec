//
// Created by hossein on 11/10/18.
//

#ifndef INC_6_1_RD_ALGOAGG_BASE_H
#define INC_6_1_RD_ALGOAGG_BASE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../Libs/fish.h"

using namespace std;
using namespace cv;

enum class QUALITY_METHODS {VOL, RankBluro, fishSharpness};

class Assessor{
public:
    Assessor(char* QUALITY_METHOD){
        this->QUALITY_METHOD = QUALITY_METHOD;
    }

    double qualityAssess(Mat image,int sample) {
        if (QUALITY_METHOD == "vol")
            return VOL_method(image);
        else if (QUALITY_METHOD == "bluro")
            return rankBluro(image);
        else if (QUALITY_METHOD == "fish")
            return fishSharpness(image,sample);
        }

private:
    char *QUALITY_METHOD;

    // Find image quality by variance of laplacian method
    double VOL_method(Mat image){
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

    // Find image quality by rank bluro method
    double rankBluro(Mat image){
        Mat temp = image;
        image.convertTo(image, CV_32F, 1 / 255.0);
        Mat gx, gy;
        Sobel(image, gx, CV_32F, 1, 0, 1);
        Sobel(image, gy, CV_32F, 0, 1, 1);
        Mat mag, angle;
        cartToPolar(gx, gy, mag, angle, true);
        double s = cv::sum(mag)[0];
        double score = s / (mag.rows * mag.cols);

        return score;
    }

    // Find image quality by fish sharpness method
    double fishSharpness(Mat &image,int sample){
        std::ostringstream streamOutput;
        streamOutput << "../output/patches/" << sample<<"_patch.png"; //TODO it is incorrect
        string imgPath = streamOutput.str();

        imwrite(imgPath,image);

        mwArray path = mwArray(imgPath.c_str());
        mwArray res;

        fish(1,res,path);

        double val = res;

        return val;
    }

};

#endif //INC_6_1_RD_ALGOAGG_BASE_H
