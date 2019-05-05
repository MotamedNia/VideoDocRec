//
// Created by hossein on 11/5/18.
//
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class assessorBase{
protected:
    int windowSize;
    int step;
    Mat scores;

    virtual double qualityAssess(Mat image) = 0;
public:
    assessorBase(int windowSize, int step){
        this->windowSize = windowSize;
        this->step = step;
    }

    Mat assess(Mat image, Mat imageMask){
        // method variables
        Mat scores = Mat::zeros(image.size(), DataType<double>::type);



        int i;
        int j;
        for ( i=0 ; i < image.cols ; i+= this->step){
            for (j =0 ; j < image.rows ; j+= this->step){
                Rect roi;
                int OWidth = image.cols -i ;
                int OHeight = image.rows -j ;

                if (i+ windowSize > image.cols && j+windowSize > image.rows)
                    roi = Rect(i,j, OWidth, OHeight);
                else if (i+ windowSize > image.cols)
                    roi = Rect(i,j, OWidth, windowSize);
                else if (j+windowSize > image.rows)
                    roi = Rect(i,j, windowSize,OHeight);
                else
                    roi = Rect(i,j, windowSize,windowSize);
                Mat win = image(roi);
                Mat win_gray = imageMask(roi); // TODO imageMask(roi);

                if (win_gray.type() == CV_8UC3)// TODO tracing required
                    cvtColor(win_gray, win_gray, COLOR_RGB2GRAY);

                int ImgNNZP = countNonZero(win_gray);

                int numOfNonZeroPixels = win.rows*win.cols - 0;

                if (ImgNNZP < numOfNonZeroPixels) {
                    continue;
                } else{
                    double qualityScore = qualityAssess(win);
                    Mat qa = (scores(roi) + qualityScore)/2;
                    scores(roi) =  qualityScore;
                    //scores(scores(roi) < qualityScore) =  qualityScore TODO implement such syntax
                }


            }
        }


        return scores;
    }

};
