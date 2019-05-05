//
// Created by hossein on 11/10/18.
//

#ifndef INC_6_1_RD_ALGOAGG_SCOREDIMAGE_H
#define INC_6_1_RD_ALGOAGG_SCOREDIMAGE_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ScoredImage{
private:
    Mat image;
    double quality;
    double similarityAverage;
public:
    ScoredImage(Mat image,double quality, double similarityAverage =0){
        this->image = image;
        this->quality = quality;
        this->similarityAverage = similarityAverage;
    }

    Mat getImage(){ return image; }
    double getQuality(){ return quality; }
    double getSimiliratyAverage(){ return similarityAverage; }

    void setImage(Mat image){
        this->image = image;
    }

    void setQuality(double quality){
        this->quality = quality;
    }

    void setSimilarityAverage(double similarityAverage){
        this->similarityAverage = similarityAverage;
    }
};
#endif //INC_6_1_RD_ALGOAGG_SCOREDIMAGE_H
