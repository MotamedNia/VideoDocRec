//
// Created by mohamad on 5/8/17.
//
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "util.h"

using namespace cv;
using namespace std;

namespace util {
//    mongocxx::v_noabi::instance inst{};
//    string db = "mongodb://172.16.0.7:27017";
//    string db = "mongodb://192.168.1.4:32769";

//    mongocxx::client conn{mongocxx::uri{db}};
//    auto builder = bsoncxx::builder::stream::document{};
//    auto collection = conn["testdb"]["ratecollection"];

//    mongocxx::collection getCollection() {
//        return collection;
//    }
    using json = nlohmann::json;

    double getMSSIM(const Mat &i1, const Mat &i2) {
        const double C1 = 6.5025, C2 = 58.5225;
        /***************************** INITS **********************************/
        int d = CV_32F;

        Mat I1, I2;
        i1.convertTo(I1, d);           // cannot calculate on one byte large values
        i2.convertTo(I2, d);

        Mat I2_2 = I2.mul(I2);        // I2^2
        Mat I1_2 = I1.mul(I1);        // I1^2
        Mat I1_I2 = I1.mul(I2);        // I1 * I2

        /***********************PRELIMINARY COMPUTING ******************************/

        Mat mu1, mu2;   //
        GaussianBlur(I1, mu1, Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, Size(11, 11), 1.5);

        Mat mu1_2 = mu1.mul(mu1);
        Mat mu2_2 = mu2.mul(mu2);
        Mat mu1_mu2 = mu1.mul(mu2);

        Mat sigma1_2, sigma2_2, sigma12;

        GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;

        GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;

        GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;

        ///////////////////////////////// FORMULA ////////////////////////////////
        Mat t1, t2, t3;

        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

        Mat ssim_map;
        divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

        Scalar mssim = mean(ssim_map); // mssim = average of ssim map

//    return (mssim[0] + mssim[1] + mssim[2]) / 3;
        return mssim[0];
    }

    static double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }
    Point2f read_out_size(string fileName) {
        std::ifstream i(fileName);
        json j;
        i >> j;
        int x = j["target_image_shape"]["x_len"];
        int y = j["target_image_shape"]["y_len"];

        return Point2f(x, y);
    }

    int read_ref_frame(string fileName) {
        std::ifstream i(fileName);
        json j;
        i >> j;
        int targetx = j["reference_frame_id"];

        return targetx;
    }

    vector<Point2f> read_points(string fileName) {
        vector<Point2f> p;
        std::ifstream i(fileName);
        json j;
        i >> j;
        double x, y;
//    for (auto &element : j["object_coord_in_ref_frame"]) {
//        sj["object_coord_in_ref_frame"]td::cout << element << '\no';
        x = j["object_coord_in_ref_frame"]["top_left"]["x"];
        y = j["object_coord_in_ref_frame"]["top_left"]["y"];
        p.push_back(Point2f(x, y));
        x = j["object_coord_in_ref_frame"]["top_right"]["x"];
        y = j["object_coord_in_ref_frame"]["top_right"]["y"];
        p.push_back(Point2f(x, y));
        x = j["object_coord_in_ref_frame"]["bottom_right"]["x"];
        y = j["object_coord_in_ref_frame"]["bottom_right"]["y"];
        p.push_back(Point2f(x, y));
        x = j["object_coord_in_ref_frame"]["bottom_left"]["x"];
        y = j["object_coord_in_ref_frame"]["bottom_left"]["y"];
        p.push_back(Point2f(x, y));
//    }
//    util::four_point_transform(imread("frame0.jpg"), p);
//    cout<<p<<endl;
        return p;
    }

    bool is_clear_warp(Mat mat) {
        Mat gray;
        cvtColor(mat, gray, COLOR_BGR2GRAY);
        threshold(gray, gray, 0, 255, THRESH_BINARY);
//    double TotalNumberOfPixels = gray.rows * gray.cols;
//    double ZeroPixels = TotalNumberOfPixels - cv::countNonZero(gray);
//    ZeroPixels = (ZeroPixels / TotalNumberOfPixels) * 100;
//    if (ZeroPixels < 10.0)
//        return false;

        vector<vector<Point> > contours;
        findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        if (contours.size() != 1)
            return false;
        vector<Point> approx;
        approxPolyDP(Mat(contours[0]), approx, arcLength(Mat(contours[0]), true) * 0.02, true);

//        if (approx.size() != 4)
//            return false;
        double maxCosine = 0;
        double minCosine = 0;

        for (int j = 2; j < 5; j++) {
            // find the maximum cosine of the angle between joint edges
            double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
            maxCosine = MAX(maxCosine, cosine);
            minCosine = MIN(maxCosine, cosine);
        }

        // if cosines of all angles are small
        // (all angles are ~90 degree) then write quandrange
        // vertices to resultant sequence
        if (maxCosine > 0.3)
            return false;
        if (minCosine < -0.3)
            return false;
        return true;
    }



    vector<Point2f> Points(vector<KeyPoint> keypoints) {
        vector<Point2f> res;
        for (unsigned i = 0; i < keypoints.size(); i++) {
            res.push_back(keypoints[i].pt);
        }
        return res;
    }

    bool niceHomography(const cv::Mat H) {
        const double det = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1);
        if (det < 0)
            return false;

        const double N1 = sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0));
        if (N1 > 4 || N1 < 0.1)
            return false;

        const double N2 = sqrt(H.at<double>(0, 1) * H.at<double>(0, 1) + H.at<double>(1, 1) * H.at<double>(1, 1));
        if (N2 > 4 || N2 < 0.1)
            return false;

        const double N3 = sqrt(H.at<double>(2, 0) * H.at<double>(2, 0) + H.at<double>(2, 1) * H.at<double>(2, 1));
        if (N3 > 0.002)
            return false;

        return true;
    }

    Mat getMean(const std::vector<Mat> &images) {
        if (images.empty()) return Mat3b();

        // Create a 0 initialized image to use as accumulator
        Mat m(images[0].rows, images[0].cols, CV_64FC3);
        m.setTo(Scalar(0, 0, 0, 0));

        // Use a temp image to hold the conversion of each input image to CV_64FC3
        // This will be allocated just the first time, since all your images have
        // the same size.
        Mat temp;
        for (int i = 0; i < images.size(); ++i) {
            // Convert the input images to CV_64FC3 ...
            images[i].convertTo(temp, CV_64FC3);

            // ... so you can accumulate
            m += temp;
        }

        // Convert back to CV_8UC3 type, applying the division to get the actual mean
        m.convertTo(m, CV_8U, 1. / images.size());
        return m;
    }


    double calculateSD(float data[]) {
        double sum = 0.0, mean, standardDeviation = 0.0;

        int i;

        for (i = 0; i < 10; ++i) {
            sum += data[i];
        }

        mean = sum / 10;

        for (i = 0; i < 10; ++i)
            standardDeviation += pow(data[i] - mean, 2);

        return sqrt(standardDeviation / 10);
    }

    bool r_sort(const frame &a, const frame &b) {
        return a.r < b.r;
    }

    bool n_sort(const frame &a, const frame &b) {
        return a.no < b.no;
    }

    bool p_sort_dec(const patch &a, const patch &b) {
        return a.rate > b.rate;
    }
    bool p_ssim_sort_dec(const patch &a, const patch &b){
        return a.MSSIM > b.MSSIM;
    }

    bool ss_sort_dec(const patch &a, const patch &b) {
        return a.MSSIM > b.MSSIM;
    }

    bool p_sort(const patch &a, const patch &b) {
        return a.rate < b.rate;
    }

    bool r_sort_dec(const frame &a, const frame &b) {
        return a.r > b.r;
    }


    bool x_sort(const Point2f &a, const Point2f &b) {
        return a.x < b.x;
    }

    bool y_sort(const Point2f &a, const Point2f &b) {
        return a.y < b.y;
    }

    double dist(Point p, Point q) {
        return std::sqrt((p.x - q.x) * (p.x - q.x) +
                         (p.y - q.y) * (p.y - q.y));
    }

//    double compareHash(cv::InputArray hashOne, cv::InputArray hashTwo) {
//        return norm(hashOne, hashTwo, NORM_HAMMING);
//    }
//
//    string computebmh(cv::InputArray inputArr, cv::OutputArray outputArr) {
////        enum {
//        int imgWidth = 256,
//                imgHeight = 256,
//                blockWidth = 16,
//                blockHeigth = 16,
//                blockPerCol = imgHeight / blockHeigth,
//                blockPerRow = imgWidth / blockWidth,
//                rowSize = imgHeight - blockHeigth,
//                colSize = imgWidth - blockWidth;
////        };
//        cv::Mat grayImg_;
//        std::vector<double> mean_;
//        cv::Mat resizeImg_;
//        cv::Mat const input = inputArr.getMat();
//        CV_Assert(input.type() == CV_8UC4 ||
//                  input.type() == CV_8UC3 ||
//                  input.type() == CV_8U);
//
//        cv::resize(input, resizeImg_, cv::Size(imgWidth, imgHeight));
//        if (input.type() == CV_8UC3) {
//            cv::cvtColor(resizeImg_, grayImg_, CV_BGR2GRAY);
//        } else if (input.type() == CV_8UC4) {
//            cv::cvtColor(resizeImg_, grayImg_, CV_BGRA2GRAY);
//        } else {
//            grayImg_ = resizeImg_;
//        }
//
//        int pixColStep = blockWidth;
//        int pixRowStep = blockHeigth;
//        int numOfBlocks = 0;
////        numOfBlocks = blockPerCol * blockPerRow;
//        pixColStep /= 2;
//        pixRowStep /= 2;
//        numOfBlocks = (blockPerCol * 2 - 1) * (blockPerRow * 2 - 1);
//
//        mean_.resize(numOfBlocks);
//        size_t blockIdx = 0;
//        for (int row = 0; row <= rowSize; row += pixRowStep) {
//            for (int col = 0; col <= colSize; col += pixColStep) {
//                mean_[blockIdx++] = cv::mean(grayImg_(cv::Rect(col, row, blockWidth, blockHeigth)))[0];
//            }
//        }
//        outputArr.create(1, numOfBlocks / 8 + numOfBlocks % 8, CV_8U);
//        cv::Mat hash = outputArr.getMat();
//        string hh = "";
//
//        double const median = cv::mean(grayImg_)[0];
//        uchar *hashPtr = hash.ptr<uchar>(0);
//        std::bitset<8> bits = 0;
//        for (size_t i = 0; i < mean_.size(); ++i) {
//            size_t const residual = i % 8;
//            bits[residual] = mean_[i] < median ? 0 : 1;
//            if (residual == 7) {
//                *hashPtr = static_cast<uchar>(bits.to_ulong());
//                ++hashPtr;
//                int t = bits.to_ulong();
//                hh.append(to_string(t));
//            } else if (i == mean_.size() - 1) {
//                *hashPtr = bits[residual];
//            }
//        }
//        return hh;
//    }
//
//
//    string computeHashph(cv::InputArray inputArr, cv::OutputArray outputArr) {
//        cv::Mat bitsImg;
//        cv::Mat dctImg;
//        cv::Mat grayFImg;
//        cv::Mat grayImg;
//        cv::Mat resizeImg;
//        cv::Mat topLeftDCT;
//        string hh = "";
//        cv::Mat const input = inputArr.getMat();
//        CV_Assert(input.type() == CV_8UC4 ||
//                  input.type() == CV_8UC3 ||
//                  input.type() == CV_8U);
//
//        cv::resize(input, resizeImg, cv::Size(32, 32));
//        if (input.type() == CV_8UC3) {
//            cv::cvtColor(resizeImg, grayImg, CV_BGR2GRAY);
//        } else if (input.type() == CV_8UC4) {
//            cv::cvtColor(resizeImg, grayImg, CV_BGRA2GRAY);
//        } else {
//            grayImg = resizeImg;
//        }
//
//        grayImg.convertTo(grayFImg, CV_32F);
//        cv::dct(grayFImg, dctImg);
//        dctImg(cv::Rect(0, 0, 8, 8)).copyTo(topLeftDCT);
//        topLeftDCT.at<float>(0, 0) = 0;
//        float const imgMean = static_cast<float>(cv::mean(topLeftDCT)[0]);
//
//        cv::compare(topLeftDCT, imgMean, bitsImg, CMP_GT);
//        bitsImg /= 255;
//        outputArr.create(1, 8, CV_8U);
//        cv::Mat hash = outputArr.getMat();
//        uchar *hash_ptr = hash.ptr<uchar>(0);
//        uchar const *bits_ptr = bitsImg.ptr<uchar>(0);
//        std::bitset<8> bits;
//        for (size_t i = 0, j = 0; i != bitsImg.total(); ++j) {
//            for (size_t k = 0; k != 8; ++k) {
//                //avoid warning C4800, casting do not work
//                bits[k] = bits_ptr[i++] != 0;
//            }
//            hash_ptr[j] = static_cast<uchar>(bits.to_ulong());
//            int t = bits.to_ulong();
//            hh.append(to_string(t));
//        }
//        return hh;
//    }

//    void updateRate(std::string hash, std::string key, double val) {
//        collection.update_one(document{} << "hash" << hash << finalize,
//                              document{} << "$set" << open_document <<
//                                         key << val << close_document << finalize,
//                              mongocxx::options::update().upsert(true));
//    }
//
//    void insert_one(bsoncxx::document::value doc) {
//        collection.insert_one(doc.view());
//    }
//
//    bsoncxx::document::value find_one(bsoncxx::document::value doc) {
//        bsoncxx::stdx::optional<bsoncxx::document::value> maybe_result =
//                collection.find_one(doc.view());
//        if (maybe_result) {
//            return maybe_result.value();
//        }
//    }

//    string computeHash(cv::InputArray inputArr, cv::OutputArray outputArr) {
//        return computeHashph(inputArr, outputArr) + computebmh(inputArr, outputArr);
//    }

    Mat resize(Mat image, int width, int height, int inter) {
// initialize the dimensions of the image to be resized and
// grab the image size
        Size dim, hw;
        hw = image.size();

// if both the width and height are None, then return the
// original image
        if (width == 0 && height == 0)
            return image;

// check to see if the width is None
        if (width == 0) {
// calculate the ratio of the height and construct the
// dimensions
            float r = float(height) / float(hw.height);
            dim = Size((hw.width * r), height);
        } else {
// calculate the ratio of the width and construct the
// dimensions
            float r = float(width) / float(hw.width);
            dim = Size(width, (hw.height * r));
        }
        Mat resized;
        resize(image, resized, dim, 0, 0, inter);

        return resized;
    }

    vector<Point2f> order_points(vector<Point2f> points) {

        //sort the points based on their x-coordinates
        std::sort(points.begin(), points.end(), x_sort);
        //grab the left-most and right-most points from the sorted x-roodinate points
        std::vector<Point2f> leftMost(points.begin(), points.begin() + points.size() / 2),
                rightMost(points.begin() + points.size() / 2, points.end());
        //now, sort the left-most coordinates according to their
        //y-coordinates so we can grab the top-left and bottom-left
        //points, respectively
        std::sort(leftMost.begin(), leftMost.end(), y_sort);
        Point2f tl = leftMost[0];
        Point2f bl = leftMost[1];
        //now, sort the right-most coordinates according to their
        //y-coordinates so we can grab the top-right and bottom-right
        //points, respectively
        std::sort(rightMost.begin(), rightMost.end(), y_sort);
        Point2f tr = rightMost[0];
        Point2f br = rightMost[1];
        //return the coordinates in top-left, top-right,bottom-right, and bottom-left order
        vector<Point2f> re_order;
        re_order.reserve(points.size());
        re_order.push_back(tl);
        re_order.push_back(tr);
        re_order.push_back(br);
        re_order.push_back(bl);
        return re_order;
    }

    Mat four_point_transform(Mat image, vector<Point2f> pts) {
        //obtain a consistent order of the points
        vector<Point2f> sorted_pts = order_points(pts);

        //compute the width of the new image, which will be the
        //maximum distance between bottom-right and bottom-left
        //x-coordiates or the top-right and top-left x-coordinates
        double widthA = dist(sorted_pts[0], sorted_pts[1]);
        double widthB = dist(sorted_pts[2], sorted_pts[3]);
        double maxWidth = max(int(widthA), int(widthB));

        //compute the height of the new image, which will be the
        //maximum distance between the top-right and bottom-right
        //y-coordinates or the top-left and bottom-left y-coordinates
        double heightA = dist(sorted_pts[0], sorted_pts[3]);
        double heightB = dist(sorted_pts[2], sorted_pts[1]);
        double maxHeight = max(int(heightA), int(heightB));

        //now that we have the dimensions of the new image, construct
        //the set of destination points to obtain a "birds eye view",
        //(i.e. top-down view) of the image, again specifying points
        //in the top-left, top-right, bottom-right, and bottom-left
        //order
        vector<Point2f> dst;
        dst.push_back(Point2f(0, 0));
        dst.push_back(Point2f(maxWidth - 1, 0));
        dst.push_back(Point2f(maxWidth - 1, maxHeight - 1));
        dst.push_back(Point2f(0, maxHeight - 1));

        //compute the util transform matrix and then apply it
        Mat M = cv::getPerspectiveTransform(sorted_pts, dst);
        Mat warped;
        cv::warpPerspective(image, warped, M, Size(maxWidth, maxHeight));
        //return the warped image
        imwrite("war.png", warped);
        return warped;
    }

    Mat four_point_transform(Mat image, vector<Point2f> sorted_pts, vector<Point2f> sorted_dst) {

        //compute the width of the new image, which will be the
        //maximum distance between bottom-right and bottom-left
        //x-coordiates or the top-right and top-left x-coordinates
        double widthA = dist(sorted_dst[0], sorted_dst[1]);
        double widthB = dist(sorted_dst[2], sorted_dst[3]);
        double maxWidth = min(int(widthA), int(widthB));

        //compute the height of the new image, which will be the
        //maximum distance between the top-right and bottom-right
        //y-coordinates or the top-left and bottom-left y-coordinates
        double heightA = dist(sorted_dst[0], sorted_dst[3]);
        double heightB = dist(sorted_dst[2], sorted_dst[1]);
        double maxHeight = min(int(heightA), int(heightB));

        //compute the util transform matrix and then apply it
        Mat M = cv::getPerspectiveTransform(sorted_pts, sorted_dst);
        Mat warped;
        cv::warpPerspective(image, warped, M, Size(maxWidth, maxHeight));
        //return the warped image
        imwrite("war.png", warped);
        return warped;
    }
    Mat four_point_transform_homo(Mat image, vector<Point2f> sorted_pts, vector<Point2f> sorted_dst) {

        //compute the width of the new image, which will be the
        //maximum distance between bottom-right and bottom-left
        //x-coordiates or the top-right and top-left x-coordinates
        double widthA = dist(sorted_dst[0], sorted_dst[1]);
        double widthB = dist(sorted_dst[2], sorted_dst[3]);
        double maxWidth = min(int(widthA), int(widthB));

        //compute the height of the new image, which will be the
        //maximum distance between the top-right and bottom-right
        //y-coordinates or the top-left and bottom-left y-coordinates
        double heightA = dist(sorted_dst[0], sorted_dst[3]);
        double heightB = dist(sorted_dst[2], sorted_dst[1]);
        double maxHeight = min(int(heightA), int(heightB));

        //compute the util transform matrix and then apply it
        Mat M = cv::getPerspectiveTransform(sorted_pts, sorted_dst);
//        Mat warped;
//        cv::warpPerspective(image, warped, M, Size(maxWidth, maxHeight));
        //return the warped image
//        imwrite("war.png", warped);
        return M;
    }
}