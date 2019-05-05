//
// Created by hossein on 11/3/18.
//
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "Libs/RobustMatcher.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

class Dewarper_CPU{
public:
    static Mat warp(Mat srcFrame, Mat inProcFrame,Mat &featureMask ,Mat &srcHomography , Size targetDimen, Mat &dewarpMask){
        // Start the timer - the time moving data between GPU and CPU is added

        vector<KeyPoint> keypoints_src, keypoints_inProc; // keypoints
        Mat descriptors_src, descriptors_inProc; // descriptors (features)

        //-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
        int minHessian = 200;
        Ptr<SURF> surf = SURF::create(minHessian);

        if (featureMask.rows == 0 || featureMask.cols == 0)
            surf->detectAndCompute( srcFrame, noArray(), keypoints_src, descriptors_src );
        else
            surf->detectAndCompute( srcFrame, featureMask, keypoints_src, descriptors_src );

        surf->detectAndCompute( inProcFrame, noArray(), keypoints_inProc, descriptors_inProc );


        //-- Step 3: Matching descriptor vectors using BruteForceMatcher
        Ptr< DescriptorMatcher > matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector< vector < DMatch> > matches12 ,matches21;

        matcher->knnMatch(descriptors_src, descriptors_inProc, matches12, 2);
        matcher->knnMatch(descriptors_inProc, descriptors_src, matches21, 2);

//        Mat drawnFeaturesImg;
//        drawKeypoints(srcFrame,keypoints_src,drawnFeaturesImg);
//        namedWindow("frame1", WINDOW_NORMAL);
//        resizeWindow("frame1", 720, 720);
//        imshow("frame1", drawnFeaturesImg);
//        waitKey(1000); TODO visualize features

        //-- Step 4: Select only goot matches
        std::vector< DMatch > good_matches;

        RobustMatcher robustMatcher;

        robustMatcher.ratioTest(matches12);
        robustMatcher.ratioTest(matches21);

        robustMatcher.symmetryTest(matches12, matches21,good_matches);

        //-- Step 6: Find Homography
        if (good_matches.size() <= 5){
            cout << "low good matches" <<endl;
            // Create new mask
            featureMask = (Mat::ones(srcFrame.size(),CV_8U))*255;
            return Mat();
        }

        Mat homography = getHomography( good_matches, keypoints_src, keypoints_inProc);

        CvMat H = CvMat(homography);
        bool isNiceHomography = true;//niceHomography(&H);
        if (homography.rows == 0 || homography.cols == 0 || !isNiceHomography) {
            cout << "bad homography" <<endl;
            // Create new mask
            featureMask = (Mat::ones(srcFrame.size(),CV_8U))*255;
            return Mat();
        }

        // Create new mask
        featureMask = (Mat::ones(srcFrame.size(),CV_8U))*255;
        warpPerspective(featureMask, featureMask,homography, srcFrame.size());

        // Dewarp image
        homography = srcHomography * homography;

        Mat dewarpedImage;

        warpPerspective(inProcFrame,dewarpedImage,homography, targetDimen);

        dewarpMask = (Mat::ones(inProcFrame.size(),CV_8U))*255;
        warpPerspective(dewarpMask, dewarpMask,homography, targetDimen);
        //-- Step 7: Show/save matches
        //imshow("Good Matches & Object detection", img_matches);
        //waitKey(0);
//        imwrite("out.png", img_matches);

        //-- Step 8: Release objects from the GPU memory
        matcher.release();
        homography.release();

        return dewarpedImage;
    }

    static Mat getHomography(const std::vector<DMatch>& good_matches,
                         const std::vector<KeyPoint>& keypoints_src,
                         const std::vector<KeyPoint>& keypoints_inProc)
    {
            //-- Localize the object
            std::vector<Point2f> src;
            std::vector<Point2f> inProc;
            for (int i = 0; i < good_matches.size(); i++) {
                    //-- Get the keypoints from the good matches
                    src.push_back(keypoints_src[good_matches[i].queryIdx].pt);
                    inProc.push_back(keypoints_inProc[good_matches[i].trainIdx].pt);
            }

            try {
                    Mat H = findHomography(inProc, src, RANSAC);

                    return H;
            } catch (Exception& e) {}
    }

    static bool niceHomography(const CvMat * H) {
        const double det = cvmGet(H, 0, 0) * cvmGet(H, 1, 1) - cvmGet(H, 1, 0) * cvmGet(H, 0, 1);
        if (det < 0)
            return false;

        const double N1 = sqrt(cvmGet(H, 0, 0) * cvmGet(H, 0, 0) + cvmGet(H, 1, 0) * cvmGet(H, 1, 0));
        if (N1 > 4 || N1 < 0.1)
            return false;

        const double N2 = sqrt(cvmGet(H, 0, 1) * cvmGet(H, 0, 1) + cvmGet(H, 1, 1) * cvmGet(H, 1, 1));
        if (N2 > 4 || N2 < 0.1)
            return false;

        const double N3 = sqrt(cvmGet(H, 2, 0) * cvmGet(H, 2, 0) + cvmGet(H, 2, 1) * cvmGet(H, 2, 1));
        if (N3 > 0.002)
            return false;

        return true;
    }
};
