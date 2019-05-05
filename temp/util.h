//
// Created by mohamad on 5/8/17.
//

#ifndef POISSONBLEND_TRANSFORM_H
#define POISSONBLEND_TRANSFORM_H

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
//#include <bsoncxx/builder/stream/document.hpp>
//#include <bsoncxx/json.hpp>
#include <bitset>

//#include <mongocxx/client.hpp>
//#include <mongocxx/instance.hpp>
#include "../src/CSVWriter.h"
#include "../src/json.hpp"

//using bsoncxx::builder::stream::close_array;
//using bsoncxx::builder::stream::close_document;
//using bsoncxx::builder::stream::document;
//using bsoncxx::builder::stream::finalize;
//using bsoncxx::builder::stream::open_array;
//using bsoncxx::builder::stream::open_document;
//
//using bsoncxx::builder::basic::kvp;
namespace util {

    struct frame {
        int no;
        double r;
    };
    struct patch {
        int no;
        double rate;
        double stdd;
        int x, y;
        double MSSIM;
        int fr;
    };


    double getMSSIM(const cv::Mat &i1, const cv::Mat &i2);
   cv:: Point2f read_out_size(std::string fileName) ;

    int read_ref_frame(std::string fileName) ;

   std:: vector<cv::Point2f> read_points(std::string fileName) ;


    std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints);

    static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);

    bool is_clear_warp(cv::Mat mat);

    bool niceHomography(const cv::Mat H);

    cv::Mat getMean(const std::vector<cv::Mat> &images);

    double calculateSD(float data[]);

    bool r_sort(const frame &a, const frame &b);

    bool n_sort(const frame &a, const frame &b);

    bool p_sort_dec(const patch &a, const patch &b);
    bool p_ssim_sort_dec(const patch &a, const patch &b);

    bool ss_sort_dec(const patch &a, const patch &b);

    bool p_sort(const patch &a, const patch &b);

    bool r_sort_dec(const frame &a, const frame &b);


    //    mongocxx::collection getCollection();
    std::vector<cv::Point2f> order_points(std::vector<cv::Point2f> points);

    cv::Mat four_point_transform(cv::Mat image, std::vector<cv::Point2f> pts);

    cv::Mat four_point_transform(cv::Mat image, std::vector<cv::Point2f> pts, std::vector<cv::Point2f> dst);
//    cv::Mat resize(cv::Mat image, int width = 0, int height = 0, int inter = cv::INTER_AREA) ;
    cv::Mat four_point_transform_homo(cv::Mat image, std::vector<cv::Point2f> sorted_pts, std::vector<cv::Point2f> sorted_dst) ;

//    std::string computebmh(cv::InputArray inputArr, cv::OutputArray outputArr);
//    std::string computeHash(cv::InputArray inputArr, cv::OutputArray outputArr);
//    void updateRate(std::string hash,std::string key,double val);
//    void insert_one(bsoncxx::document::value doc);
//    bsoncxx::document::value find_one(bsoncxx::document::value doc);
};


#endif //POISSONBLEND_TRANSFORM_H
