//
// Created by hossein on 11/10/18.
//
#include <iostream>
#include "Assessor.h"
#include <dirent.h>

using namespace std;

class Composer{
private:
    Mat scoredImages;
    Mat scores;
    Mat dewarpedRefFrame;
    int windowSize;
    int step;
    bool isGPU;
//    QUALITY_METHODS method = QUALITY_METHODS ::fishSharpness;
    Assessor assessor;

public:
    Composer(int windowSize,Mat dewarpedRefFrame, Mat scoredImages, Mat scores,char *quality, bool isGPU):assessor(quality){
        this->scoredImages = scoredImages;
        this->scores = scores;
        this->dewarpedRefFrame = dewarpedRefFrame;
        this->windowSize = windowSize;
        this->step = windowSize/2;
        this->isGPU = isGPU;
    }

    void compose(Mat image,Mat imageMask,int frame_num,int sample){
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
                Mat win_mask = imageMask(roi);

                if (win_mask.type() == CV_8UC3)
                    cvtColor(win_mask, win_mask, COLOR_RGB2GRAY);

                int ImgNNZP = countNonZero(win_mask);

                int numOfNonZeroPixels = win.rows*win.cols - 0;

                if (ImgNNZP < numOfNonZeroPixels) {
                    continue;
                } else{
                    Mat refWin = dewarpedRefFrame(roi);

                    double qualityScore = assessor.qualityAssess(win,sample);
                    double refQualityScore = assessor.qualityAssess(refWin,sample);

//                    if (i ==384 && j == 128 && frame_num > 36)
//                        cout << qualityScore << " :: " << refQualityScore << "   "; // TODO tracing

                    if (qualityScore >= refQualityScore){
                        Scalar ssim;
                        if (isGPU)
                            ssim = getMSSIM_CUDA(refWin, win);
                        else
                            ssim = getMSSIM(refWin, win);

                        double ssim_av = (ssim.val[0]+ssim.val[1]+ssim.val[2])/3;
                        if (ssim_av > 0.5){
                            Mat dewarpedWin;
                            if (win.rows < 256 || win.cols <256)
                                dewarpedWin = win;
                            else
                                dewarpedWin = win;// TODO warp(refWin,win); it's not dewarp correctly

                            if (dewarpedWin.rows == 0 || dewarpedWin.cols == 0)
                                continue;
                            else{
                                // region TODO visualization by saving program data
                                // create patch directory
//                               {
//                                   std::ostringstream patchDir;
//                                   patchDir << "mkdir -p /media/hossein/main_repo/ICDAR2017/Implementations/main/6.1_AlgoAgg_outputs/output/" << sample << "/" << i << "_" << j;
//                                //    patchDir << "mkdir -p ../output/" << sample << "/" << i << "_" << j;
//                                   string outputFilePath = patchDir.str();
//                                   system(outputFilePath.c_str());
//                               }
//
////                                {
//                                   std::ostringstream imgDir;
//                                   imgDir << "/media/hossein/main_repo/ICDAR2017/Implementations/main/6.1_AlgoAgg_outputs/output/" << sample << "/" << i << "_" << j <<"/"<<frame_num <<
//                                   "_" << qualityScore <<".jpg";
//                            //    imgDir << "../output/" << sample << "/" << i << "_" << j <<"/"<<frame_num <<
//                            //           "_" << qualityScore <<".jpg";
//                                   string outputPatchPath = imgDir.str();
//                                   imwrite(outputPatchPath, dewarpedWin);
////                                } TODO commented temporally

                                // endregion

                                Mat doubleImg;
                                dewarpedWin.convertTo(doubleImg,DataType<double>::type);
                                multiply(doubleImg ,Scalar(qualityScore,qualityScore,qualityScore),doubleImg);//#########
                                this->scoredImages(roi) +=  doubleImg;

                                this->scores(roi) += Scalar(qualityScore,qualityScore,qualityScore);

                            }
                        }
                    }

                    //scores(scores(roi) < qualityScore) =  qualityScore TODO implement such syntax
                }


            }
        }
    }

    Mat getScoredImages(){ return this->scoredImages; }
    Mat getScores(){ return this->scores; }

private:
    Scalar getMSSIM_CUDA( const Mat& i1, const Mat& i2) // TODO there is another function named getMSSIM_CUDA_Optimized
    {
        const float C1 = 6.5025f, C2 = 58.5225f;
        /***************************** INITS **********************************/
        cuda::GpuMat gI1, gI2, gs1, tmp1,tmp2;
        gI1.upload(i1);
        gI2.upload(i2);
        gI1.convertTo(tmp1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
        gI2.convertTo(tmp2, CV_MAKE_TYPE(CV_32F, gI2.channels()));
        vector<cuda::GpuMat> vI1, vI2;
        cuda::split(tmp1, vI1);
        cuda::split(tmp2, vI2);
        Scalar mssim;
        Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(vI2[0].type(), -1, Size(11, 11), 1.5);
        for( int i = 0; i < gI1.channels(); ++i )
        {
            cuda::GpuMat I2_2, I1_2, I1_I2;
            cuda::multiply(vI2[i], vI2[i], I2_2);        // I2^2
            cuda::multiply(vI1[i], vI1[i], I1_2);        // I1^2
            cuda::multiply(vI1[i], vI2[i], I1_I2);       // I1 * I2
            /*************************** END INITS **********************************/
            cuda::GpuMat mu1, mu2;   // PRELIMINARY COMPUTING
            gauss->apply(vI1[i], mu1);
            gauss->apply(vI2[i], mu2);
            cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
            cuda::multiply(mu1, mu1, mu1_2);
            cuda::multiply(mu2, mu2, mu2_2);
            cuda::multiply(mu1, mu2, mu1_mu2);
            cuda::GpuMat sigma1_2, sigma2_2, sigma12;
            gauss->apply(I1_2, sigma1_2);
            cuda::subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;
            gauss->apply(I2_2, sigma2_2);
            cuda::subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;
            gauss->apply(I1_I2, sigma12);
            cuda::subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;
            cuda::GpuMat t1, t2, t3;
            mu1_mu2.convertTo(t1, -1, 2, C1); // t1 = 2 * mu1_mu2 + C1;
            sigma12.convertTo(t2, -1, 2, C2); // t2 = 2 * sigma12 + C2;
            cuda::multiply(t1, t2, t3);        // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
            cuda::addWeighted(mu1_2, 1.0, mu2_2, 1.0, C1, t1);       // t1 = mu1_2 + mu2_2 + C1;
            cuda::addWeighted(sigma1_2, 1.0, sigma2_2, 1.0, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
            cuda::multiply(t1, t2, t1);                              // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
            cuda::GpuMat ssim_map;
            cuda::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
            Scalar s = cuda::sum(ssim_map);
            mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);
        }
        return mssim;
    }

    Scalar getMSSIM( const Mat& i1, const Mat& i2)
    {
        const double C1 = 6.5025, C2 = 58.5225;
        /***************************** INITS **********************************/
        int d     = CV_32F;
        Mat I1, I2;
        i1.convertTo(I1, d);           // cannot calculate on one byte large values
        i2.convertTo(I2, d);
        Mat I2_2   = I2.mul(I2);        // I2^2
        Mat I1_2   = I1.mul(I1);        // I1^2
        Mat I1_I2  = I1.mul(I2);        // I1 * I2
        /*************************** END INITS **********************************/
        Mat mu1, mu2;   // PRELIMINARY COMPUTING
        GaussianBlur(I1, mu1, Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, Size(11, 11), 1.5);
        Mat mu1_2   =   mu1.mul(mu1);
        Mat mu2_2   =   mu2.mul(mu2);
        Mat mu1_mu2 =   mu1.mul(mu2);
        Mat sigma1_2, sigma2_2, sigma12;
        GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        Mat t1, t2, t3;
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        Mat ssim_map;
        divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
        Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
        return mssim;
    }

    static Mat warp(Mat refFrame, Mat frame){ // Dewarp frame to refFrame plane

        // Copy the image into GPU memory
        cuda::GpuMat img_ref_Gpu( refFrame );
        cuda::GpuMat img_frame_Gpu( frame );

        // Start the timer - the time moving data between GPU and CPU is added

        cuda::GpuMat keypoints_ref_Gpu, keypoints_frame_Gpu; // keypoints
        cuda::GpuMat descriptors_ref_Gpu, descriptors_frame_Gpu; // descriptors (features)

        //-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
        int minHessian = 50;
        cuda::SURF_CUDA surf( minHessian );

        surf( img_ref_Gpu, cuda::GpuMat(), keypoints_ref_Gpu, descriptors_ref_Gpu );
        surf( img_frame_Gpu, cuda::GpuMat(), keypoints_frame_Gpu, descriptors_frame_Gpu );

        //-- Step 3: Matching descriptor vectors using BruteForceMatcher
        Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
        vector< vector < DMatch> > matches12 ,matches21;

        matcher->knnMatch(descriptors_ref_Gpu, descriptors_frame_Gpu, matches12, 2);
        matcher->knnMatch(descriptors_frame_Gpu, descriptors_ref_Gpu, matches21, 2);

        // Downloading results  Gpu -> Cpu
        vector< KeyPoint > keypoints_ref, keypoints_frame;

        surf.downloadKeypoints(keypoints_ref_Gpu, keypoints_ref);
        surf.downloadKeypoints(keypoints_frame_Gpu, keypoints_frame);


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
        if (good_matches.size() <= 5)
            return Mat();


        Mat homography = getHomography( good_matches, keypoints_ref, keypoints_frame);

        if (homography.rows == 0 || homography.cols == 0 )
            return Mat();

        // Dewarp image
        Mat dewarpedImage;

        warpPerspective(frame,dewarpedImage,homography, refFrame.size());

        //-- Step 7: Show/save matches
        //imshow("Good Matches & Object detection", img_matches);
        //waitKey(0);
//        imwrite("out.png", img_matches);

        //-- Step 8: Release objects from the GPU memory
        surf.releaseMemory();
        matcher.release();
        img_ref_Gpu.release();
        img_frame_Gpu.release();
        homography.release();

        return dewarpedImage;
    }

    static Mat getHomography(const std::vector<DMatch>& good_matches,
                             const std::vector<KeyPoint>& keypoints_src,
                             const std::vector<KeyPoint>& keypoints_inProc) {
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
};