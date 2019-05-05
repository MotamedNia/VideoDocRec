//
// Created by hossein on 11/3/18.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Libs/RobustMatcher.h"

using namespace std;
using namespace cv;

class Dewarper{
public:
    static Mat warp(Mat srcFrame, Mat inProcFrame,Mat &featureMask ,Mat &srcHomography , Size targetDimen, Mat &dewarpMask){

        Mat RGBInProcFrame = inProcFrame.clone();

        if (inProcFrame.type() == CV_8UC3)
            cvtColor(inProcFrame, inProcFrame, COLOR_RGB2GRAY);
        if (srcFrame.type() == CV_8UC3)
            cvtColor(srcFrame, srcFrame, COLOR_RGB2GRAY);
        // Copy the image into GPU memory
        cuda::GpuMat img_src_Gpu( srcFrame );
        cuda::GpuMat img_inProc_Gpu( inProcFrame );
        cuda::GpuMat featureMask_Gpu(featureMask);

        // Start the timer - the time moving data between GPU and CPU is added

        cuda::GpuMat keypoints_src_Gpu, keypoints_inProc_Gpu; // keypoints
        cuda::GpuMat descriptors_src_Gpu, descriptors_inProc_Gpu; // descriptors (features)

        //-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
        int minHessian = 50;
        cuda::SURF_CUDA surf( minHessian );

        if (featureMask.rows == 0 || featureMask.cols == 0)
            surf( img_src_Gpu, cuda::GpuMat(), keypoints_src_Gpu, descriptors_src_Gpu );
        else
            surf( img_src_Gpu, featureMask_Gpu, keypoints_src_Gpu, descriptors_src_Gpu );

        surf( img_inProc_Gpu, cuda::GpuMat(), keypoints_inProc_Gpu, descriptors_inProc_Gpu );
        //cout << "FOUND " << keypoints_object_Gpu.cols << " keypoints on object image" << endl;
        //cout << "Found " << keypoints_scene_Gpu.cols << " keypoints on scene image" << endl;

        //-- Step 3: Matching descriptor vectors using BruteForceMatcher
        Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
        vector< vector < DMatch> > matches12 ,matches21;

        matcher->knnMatch(descriptors_src_Gpu, descriptors_inProc_Gpu, matches12, 2);
        matcher->knnMatch(descriptors_inProc_Gpu, descriptors_src_Gpu, matches21, 2);

        // Downloading results  Gpu -> Cpu
        vector< KeyPoint > keypoints_src, keypoints_inProc;
        //vector< float> descriptors_scene, descriptors_object;
        surf.downloadKeypoints(keypoints_src_Gpu, keypoints_src);
        surf.downloadKeypoints(keypoints_inProc_Gpu, keypoints_inProc);
        //surf.downloadDescriptors(descriptors_scene_Gpu, descriptors_scene);
        //surf.downloadDescriptors(descriptors_object_Gpu, descriptors_object);

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

//        for (int k = 0; k < std::min(keypoints_src.size()-1, matches.size()); k++)
//        {
//            if ( (matches[k][0].distance < 0.75*(matches[k][1].distance)) &&
//                 ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
//            {
//                // take the first result only if its distance is smaller than 0.6*second_best_dist
//                // that means this descriptor is ignored if the second distance is bigger or of similar
//                good_matches.push_back(matches[k][0]);
//            }
//        } TODO old version before symmetry test

        //-- Step 6: Find Homography
        if (good_matches.size() <= 5){
            cout << "low good matches" <<endl;
            // Create new mask
            featureMask = (Mat::ones(srcFrame.size(),CV_8U))*255;
            return Mat();
        }

        Mat homography = getHomography( good_matches, keypoints_src, keypoints_inProc);

        // Check whether homography is correct or not
        bool isNiceHomography = niceHomography(homography); //true;//
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


        cuda::GpuMat dewarpedImage_gpu;
        cuda::GpuMat RGBinProcFrame_gpu(RGBInProcFrame);

        cuda::warpPerspective(RGBinProcFrame_gpu,dewarpedImage_gpu,homography,targetDimen);
        Mat dewarpedImage;
        dewarpedImage_gpu.download(dewarpedImage);

        dewarpMask = (Mat::ones(inProcFrame.size(),CV_8U))*255;
        warpPerspective(dewarpMask, dewarpMask,homography, targetDimen);

        // Check whether warp is correct or not
        bool isClearWarp = is_clear_warp(dewarpMask);
        if (!isClearWarp) {
            cout << "bad homography" <<endl;
            // Create new mask
            featureMask = (Mat::ones(srcFrame.size(),CV_8U))*255;
            return Mat();
        }


        //-- Step 8: Release objects from the GPU memory
        surf.releaseMemory();
        matcher.release();
        img_src_Gpu.release();
        img_inProc_Gpu.release();
        homography.release();
        dewarpedImage_gpu.release();
        RGBinProcFrame_gpu.release();

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

private:
    static bool niceHomography(const cv::Mat H) {
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

    static bool is_clear_warp(Mat mat) {
        Mat gray = mat;
        if(mat.type() == CV_8UC3)
            cvtColor(mat, gray, COLOR_BGR2GRAY);

        threshold(gray, gray, 0, 255, THRESH_BINARY);

        vector<vector<Point> > contours;
        findContours(gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        if (contours.size() != 1)
            return false;
        vector<Point> approx;
        approxPolyDP(Mat(contours[0]), approx, arcLength(Mat(contours[0]), true) * 0.02, true);


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

    static double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }
};