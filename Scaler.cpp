//
// Created by hossein on 11/3/18.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "json.hpp"

using namespace std;
using namespace cv;
using json = nlohmann::json;

class Scaler{
private:
    json data;

    int target_x;
    int target_y;
    int frame_y;
    int frame_x;

    vector<Point2f> getPaperCorners(){
        int top_r_x = data["object_coord_in_ref_frame"]["top_right"]["x"];
        int top_r_y = data["object_coord_in_ref_frame"]["top_right"]["y"];
        int top_l_x = data["object_coord_in_ref_frame"]["top_left"]["x"];
        int top_l_y = data["object_coord_in_ref_frame"]["top_left"]["y"];
        int bot_r_x = data["object_coord_in_ref_frame"]["bottom_right"]["x"];
        int bot_r_y = data["object_coord_in_ref_frame"]["bottom_right"]["y"];
        int bot_l_x = data["object_coord_in_ref_frame"]["bottom_left"]["x"];
        int bot_l_y = data["object_coord_in_ref_frame"]["bottom_left"]["y"];

        vector<Point2f> corners;
        corners.push_back(Point2f(top_l_x,top_l_y));
        corners.push_back(Point2f(top_r_x,top_r_y));
        corners.push_back(Point2f(bot_r_x,bot_r_y));
        corners.push_back(Point2f(bot_l_x,bot_l_y));


        return corners;
    }


    Mat fourPointTransform(vector<Point2f> corners, Size targetDimen, Point2f frameDimen) {
        vector<Point2f> dst;
        dst.push_back(Point2f(0, 0));
        dst.push_back(Point2f(targetDimen.width - 1, 0));
        dst.push_back(Point2f(targetDimen.width - 1, targetDimen.height - 1));
        dst.push_back(Point2f(0, targetDimen.height - 1));

        Mat homography = getPerspectiveTransform(corners, dst);

        return homography;
    }


public:
    // Constructor
    Scaler(String filePath){
        ifstream i(filePath);
        i >> data;

        this->frame_x = data["input_video_shape"]["x_len"];
        this->frame_y = data["input_video_shape"]["y_len"];
        this->target_x = data["target_image_shape"]["x_len"];
        this->target_y = data["target_image_shape"]["y_len"];
    }

    void getHomoAndDimen(Mat &homography, Size &targetDimen){
        vector<Point2f> corners = this->getPaperCorners();

        targetDimen = Size(this->target_x, this->target_y);
        Point2f frameDimen = Point2f(this->frame_x, this->frame_y);
        homography = this->fourPointTransform(corners, targetDimen, frameDimen);
    }

};

