#include "../src/fish/fish.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <blend/blend.h>
#include "../src/util.h"
#include "utility"
#include "opencv2/imgproc.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <blend/clone.h>
#include <libgen.h>
#include <zconf.h>
#include <sys/param.h>
#include "../src/util/parser.hpp"
#include <algorithm>
#include <memory>
#include "../src/kvld/kvld.h"
#include "../src/kvld/convert.h"
#include <thread>
#include "../src/RMatcher.h"

using namespace std;
using namespace cv;
using util::frame;
using util::patch;

string fullpath = "";
CSVWriter writer("rate.csv");
using namespace aria::csv;
Mat image_ref, image_nowarp;
int stride = 2;
int nfeatures = 0, nOctaveLayers = 3;
double contrastThreshold = 0.05, edgeThreshold = 10,
        sigma = 1.5;
vector<frame> rates;
bool debug = false;

map<pair<int, int>, vector<vector<uchar> > > all_rois;

double rank_blur_fish(Mat img, int i);

Mat patching(Mat mat);

Mat
FindHomoRobust(Mat ref, Mat frame, Mat ref_mask);

void add_to_rois(Mat warped, Mat fmask);

void run_kvld(Mat mat, Mat frame, vector<DMatch> matches, std::vector<KeyPoint> kpts1, std::vector<KeyPoint> kpts2,
              std::vector<DMatch> &result);


vector<frame> rateVideoB(VideoCapture &cap) {
    Mat fram;
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    if (FILE *file = fopen("rate.csv", "r")) {
        fclose(file);
        std::ifstream f("rate.csv");
        CsvParser parser = CsvParser(f);
        double s;
        int t = 0;
        for (auto &row : parser) {
            string field = row.at(2);
            s = atof(field.c_str());
            cout << s << endl;
            frame f;
            f.r = s;
            f.no = t++;
            rates.push_back(f);
        }
        cout << "parallel done!\n";
        return rates;
    }

    int c = 0;
    for (int t = 0;; t++) {
        cap.read(fram);
        if (fram.empty()) {
            cerr << "blank frame grabbed\n";
            break;
        }
        c++;
        double d = rank_blur_fish(fram, t);
        double arr[] = {0, 0.0 + t, d};
        writer.addDatainRow(arr, arr + sizeof(arr) / sizeof(double));
    }
    cout << "parallel done!\n";
    cout << c << endl << rates.size() << endl;
    fram.release();
    return rates;
}


Mat
FindHomoRobust(Mat ref, vector<KeyPoint> kpts1, Mat desc1, Mat frame, int i) {

    RMatcher rMatcher;

    int ry = frame.rows, cx = frame.cols, split, xoy;
    xoy = ry > cx ? 0 : 1;
    split = xoy ? cx / 2 : ry / 2;

    cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(/*nfeatures, nOctaveLayers, contrastThreshold,
                                                                edgeThreshold, sigma*/);
    cv::Ptr<cv::DescriptorMatcher> matcher =
            DescriptorMatcher::create("BruteForce");

//    cvtColor(frame, frame, CV_BGR2GRAY);
    vector<KeyPoint> kpts2;
    Mat desc2;
    sift->detect(frame, kpts2);
    sift->compute(frame, kpts2, desc2);

    std::vector<std::vector<cv::DMatch> > matches12, matches21;

    matcher->knnMatch(desc2, desc1, matches21, 2); // return 2 nearest neighbours

    matcher->knnMatch(desc1, desc2, matches12, 2); // return 2 nearest neighbours


    Mat homography;

    std::vector<DMatch> good_matches, good_matches1, good_matches2, matches, matches1, matches2;

    rMatcher.ratioTest(matches21);
    rMatcher.ratioTest(matches12);

    rMatcher.symmetryTest(matches21, matches12, matches);

    for (auto item:matches) {
        auto p = kpts2[item.queryIdx].pt;
        if (xoy) {
            if (p.x < split)
                matches1.push_back(item);
            else
                matches2.push_back(item);
        } else {
            if (p.y < split)
                matches1.push_back(item);
            else
                matches2.push_back(item);
        }
    }


//    std::cout << "starting first helper...\n";
    if (matches1.size()) {
        std::thread helper1(run_kvld, ref, frame, matches1, kpts2, kpts1, std::ref(good_matches1));
        helper1.join();
    }
//    std::cout << "starting second helper...\n";
    if (matches2.size()) {
        std::thread helper2(run_kvld, ref, frame, matches2, kpts2, kpts1, std::ref(good_matches2));
        helper2.join();
    }


//    std::cout << "done!\n";

//    run_kvld(ref, frame, matches1, kpts2, kpts1, good_matches1);
//    run_kvld(ref, frame, matches2, kpts2, kpts1, good_matches2);

    cout << kpts1.size() << "," << kpts2.size() << ",good:" << good_matches1.size() << ",good 2:"
         << good_matches2.size() << endl;

    if (good_matches1.size() > 8)
        for (auto item:good_matches1)
            good_matches.push_back(item);
    else
        for (auto item:matches1)
            good_matches.push_back(item);

    if (good_matches2.size() > 8)
        for (auto item:good_matches2)
            good_matches.push_back(item);
    else
        for (auto item:matches2)
            good_matches.push_back(item);

    Mat res;

    cv::drawMatches(frame, kpts2, ref, kpts1, good_matches, res, Scalar::all(-1),
                    Scalar::all(-1),
                    std::vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("bmp/" + to_string(i) + "_" + to_string(kpts1.size()) + "-" + to_string(kpts2.size()) + "res.jpg", res);

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (int i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        obj.push_back(kpts2[good_matches[i].queryIdx].pt);
        scene.push_back(kpts1[good_matches[i].trainIdx].pt);
    }

    if (good_matches.size() > 15) {
        homography = findHomography(obj, scene, RANSAC);
    }


    return homography;
}

void
run_kvld(Mat image1, Mat image2, std::vector<DMatch> matches, std::vector<KeyPoint> feat1, std::vector<KeyPoint> feat2,
         std::vector<DMatch> &result) {

    std::vector<DMatch> matchesFiltered;

    //=============== convert openCV sturctures to KVLD recognized elements
    Image<float> If1, If2;
    Convert_image(image1, If1);
    Convert_image(image2, If2);

    //===============================  KVLD method ==================================//
    std::cout << "K-VLD starts with " << matches.size() << " matches" << std::endl;

    std::vector<double> vec_score;

    //In order to illustrate the gvld(or vld)-consistant neighbors, the following two parameters has been externalized as inputs of the function KVLD.
    libNumerics::matrix<float> E = libNumerics::matrix<float>::ones(matches.size(), matches.size()) * (-1);
    // gvld-consistency matrix, intitialized to -1,  >0 consistency value, -1=unknow, -2=false

    std::vector<bool> valide(matches.size(),
                             true);// indices of match in the initial matches, if true at the end of KVLD, a match is kept.

    size_t it_num = 0;
    KvldParameters kvldparameters;//initial parameters of KVLD

    while (it_num < 5 && kvldparameters.inlierRate >
                         KVLD(If1, If2, feat1, feat2, matches, matchesFiltered, vec_score, E, valide, kvldparameters)) {
        kvldparameters.inlierRate /= 2;
        kvldparameters.rang_ratio = sqrt(2.0f);
        std::cout << "low inlier rate, re-select matches with new rate=" << kvldparameters.inlierRate << std::endl;
        if (matchesFiltered.size() == 0) kvldparameters.K = 2;
        it_num++;
    }

    for (auto item:matchesFiltered)
        result.push_back(item);

    std::cout << "K-VLD filter ends with " << matchesFiltered.size() << " selected matches" << std::endl;
}


void process(VideoCapture &capture, Mat cimg1, Mat HOMO) {
    system("mkdir bmp");
    Mat img1_clone/*, img1_he*/;
    Mat ref_mask = Mat(cimg1.size(), CV_8UC1, Scalar(255));

    img1_clone = cimg1.clone();

    vector<frame> ratevideo = rateVideoB(capture);
    cout << "rating done!" << endl;
    sort(ratevideo.begin(), ratevideo.end(), util::n_sort);
    vector<frame> sort_ratevideo(ratevideo);
    std::sort(sort_ratevideo.begin(), sort_ratevideo.end(), util::r_sort);

    capture.set(CV_CAP_PROP_POS_FRAMES, 1);

    double rate_thresh = max(3.0, sort_ratevideo[sort_ratevideo.size() / 2].r);

    vector<frame> selected_frames;
    selected_frames.reserve(888);
    int sc = ratevideo.size() / 150;
    sc = MAX(sc, 2);
    for (int c = 0; c < ratevideo.size(); c += sc) {
        int size = ratevideo.size();
        vector<frame> chunk(ratevideo.begin() + c, ratevideo.begin() + c + std::min(sc, size - c));
        std::sort(chunk.begin(), chunk.end(), util::r_sort_dec);
        frame last = chunk[0];
//        frame last = ratevideo[c];
        if (last.r > rate_thresh) {
            selected_frames.push_back(last);
            cout << "frame:" << last.no << " selected" << endl;
        } else {
            cout << "frame " << last.no << " rejected.rate:" << last.r << endl;
        }

    }
    ratevideo.clear();
    ratevideo.shrink_to_fit();
    rates.clear();
    rates.shrink_to_fit();


    vector<KeyPoint> keypoints_scene;
    Mat /*img_scene_Gray,*/ desc1;
//    cvtColor(cimg1, img_scene_Gray, CV_BGR2GRAY);

    cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(/*nfeatures*2, nOctaveLayers, contrastThreshold,
                                                                edgeThreshold, sigma*/);
    sift->detect(image_nowarp, keypoints_scene);
    sift->compute(image_nowarp, keypoints_scene, desc1);


    int i = 0;
    for (auto &s_frame : selected_frames) {
        capture.set(CV_CAP_PROP_POS_FRAMES, s_frame.no);
        i = s_frame.no;
        Mat cimg2;

        capture >> cimg2;
        if (cimg2.empty())
            break;
        cout << "frame :" << i << endl;

//        transpose(cimg2, cimg2);
//        flip(cimg2, cimg2, 1);

        Mat new_mask = Mat(cimg2.size(), CV_8UC1, Scalar(255));

        sift->detect(image_nowarp, keypoints_scene, ref_mask);
        sift->compute(image_nowarp, keypoints_scene, desc1);

        ref_mask = Mat(image_nowarp.size(), CV_8UC1, Scalar(255));

        Mat homography = FindHomoRobust(image_nowarp, keypoints_scene, desc1, cimg2, i);
        if (homography.empty()) {
            continue;
        }
//        cout<<HOMO<<endl;

//        cout<<homography<<endl;
        homography = HOMO * homography;
//        cout<<homography<<endl;

        cout << "start dewarping" << endl;
        Mat wimg2/*, mask*/;
//        warpPerspective(cimg2, wimg2, homography, image_nowarp.size(), 2);
//        cv::imwrite("bmp/" + to_string(i) + "qwimg.jpg", wimg2);

        warpPerspective(cimg2, wimg2, homography, cimg1.size(), 2);

        if (util::is_clear_warp(wimg2) && util::niceHomography(homography)) {
            cout << "check dewarping" << endl;

            cv::warpPerspective(new_mask, ref_mask, homography, cimg1.size(), 2);
//            imwrite("m.png", ref_mask);
            cv::imwrite("bmp/" + to_string(i) + "wimg.jpg", wimg2);

            add_to_rois(wimg2, ref_mask);
//            cv::imwrite("bmp/" + to_string(i) + "ref_mask.jpg", ref_mask);
            int erosion_size = 200;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_DILATE,
                                                       cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                       cv::Point(erosion_size, erosion_size));
            dilate(ref_mask, ref_mask, kernel);
            imwrite("m.png", ref_mask);

        } else {
            string str = "warp";
            if (util::niceHomography(homography))
                str = "nice";
            cv::imwrite("bmp/" + to_string(i) + str + "wimg.jpg", wimg2);
            cout << "frame :" << to_string(i) + " rejected " << endl;
        }
        wimg2.release();
    }

    cimg1 = patching(img1_clone);

    // cv::imwrite("seamless-blend2.png", cimg1);

}

void add_to_rois(Mat warped, Mat fmask) {
    cout << "add to rois start" << endl;
    Size wsize;
    int cols = warped.cols - 8;
    int rows = warped.rows - 8;
    int ox = 11, oy = 11;
    int div = 18;
    if (rows > cols) {
        wsize.height = rows / div;
        wsize.width = rows / div;
        oy = wsize.height - (wsize.height / 15);
        int tn = cols / wsize.width;
        ox = wsize.width - (wsize.width - (cols - (wsize.width * tn))) / tn;
    } else {
        wsize.height = cols / div;
        wsize.width = cols / div;
        int tn = rows / wsize.height;
        ox = wsize.width - (wsize.width / 15);
        oy = wsize.height - (wsize.height - (rows - (wsize.width * tn))) / tn;
    }


    int cnt = 0;
    bool ti = true;
    for (int i = 4; i <= warped.cols - wsize.width;) {
        bool tj = true;
        for (int j = 4; j <= warped.rows - wsize.height;) {
//            cout << "adding " << i << "," << j << endl;
            Rect r(i, j, wsize.width, wsize.height);
            Mat roi = warped(r);
            Mat mroi = fmask(r);
            double min, max;
            cv::minMaxLoc(mroi, &min, &max);
//            int nz = cv::countNonZero(mroi);
            vector<uchar> buff;
            vector<int> param = vector<int>(2);
            Mat groi;
            cv::cvtColor(roi, groi, COLOR_BGR2GRAY);
            Mat rroi = image_ref(r);
            cv::cvtColor(rroi, rroi, COLOR_BGR2GRAY);

            double ssim = util::getMSSIM(rroi, groi);

//            if (min < 100 && nz > (wsize.height * wsize.width) / 2) {
//                Mat dst,froi;
//                bitwise_not(mroi, dst);
//                froi=dst.clone();
//                Mat rroi = image_ref(r);
//                rroi.copyTo(froi, dst);
//                blend::seamlessBlend(froi, roi, dst, roi);
//                imwrite("bmp/p"+to_string(i)+'-'+to_string(j)+".jpg",roi);
//            }

            Scalar mean, stdDev;
            cv::meanStdDev(roi, mean, stdDev);

            if (min > 100 && stdDev[0] > 2 && ssim > 0.62) {
                cnt++;
                param[0] = IMWRITE_PNG_COMPRESSION;
                param[1] = 2;//default(3)  0-9.
                imencode(".png", roi, buff, param);
                all_rois[pair<int, int>(i, j)].push_back(buff);
            }


            j += oy / stride;
            if (j > warped.rows - wsize.height && tj) {
                tj = false;
                j = std::min(j, warped.rows - wsize.height) - 2;
            }
        }
        i += ox / stride;
        if (i > warped.cols - wsize.width && ti) {
            ti = false;
            i = min(i, warped.cols - wsize.width) - 2;
        }
    }
    cout << cnt << " patch added to rois" << endl;
}

Mat patching(Mat img1_clone) {

    cout << "--start patching" << endl;
    Size wsize;
    int cols = img1_clone.cols - 8;
    int rows = img1_clone.rows - 8;
    int ox = 11, oy = 11;
    int div = 18;
    if (rows > cols) {
        wsize.height = rows / div;
        wsize.width = rows / div;
        oy = wsize.height - (wsize.height / 15);
        int tn = cols / wsize.width;
        ox = wsize.width - (wsize.width - (cols - (wsize.width * tn))) / tn;
    } else {
        wsize.height = cols / div;
        wsize.width = cols / div;
        int tn = rows / wsize.height;
        ox = wsize.width - (wsize.width / 15);
        oy = wsize.height - (wsize.height - (rows - (wsize.width * tn))) / tn;
    }

    bool ti = true;

    for (int i = 4; i <= img1_clone.cols - wsize.width;) {
        bool tj = true;
        for (int j = 4; j <= img1_clone.rows - wsize.height;) {
            cout << "patching at " << i << "," << j << endl;

            std::vector<cv::Mat> rois;
            vector<patch> ptchs;
            vector<patch> sel_ptchs;
            int t = 0;
            Mat ref = img1_clone(Rect(i, j, wsize.width, wsize.height));
            cv::cvtColor(ref, ref, COLOR_BGR2GRAY);

            int pos = 0;
            for (vector<uchar> buff:all_rois[pair<int, int>(i, j)]) {
                Mat roi = imdecode(Mat(buff), CV_LOAD_IMAGE_COLOR);
                if (roi.empty())continue;
                Mat groi;
                cv::cvtColor(roi, groi, COLOR_BGR2GRAY);

                if (200) {

                    patch p;
                    p.fr = pos++;
                    p.no = t;
                    p.x = i;
                    p.y = j;
//                    p.stdd = stdDev[0];
                    p.MSSIM = util::getMSSIM(ref, groi);
//                    if (p.MSSIM < 0.6 && rstdDev[0] > 2)
//                        continue;

//                    p.rate = rank_blur(roi);

                    rois.push_back(roi);
                    ptchs.push_back(p);

                    t++;

                } else {
//                    cout << "at " << to_string(i) + "-" + to_string(j) + "-" << pos << " rejected!" << endl;
                }
            }

            all_rois[pair<int, int>(i, j)].clear();
            all_rois[pair<int, int>(i, j)].shrink_to_fit();
            if (debug) {
                string str = "mkdir bmp/" + to_string(i) + "-" + to_string(j);
                system(str.c_str());
            }
            if (ptchs.size() > 0) {
//                Mat arr[t];
//                std::copy(rois.begin(), rois.end(), arr);

                cout << "ranking patches" << endl;
//                ParallelApplyFoo(arr, t/*-1, prates*/);
                for (int t = 0; t < rois.size(); t++) {
                    rank_blur_fish(rois.at(t), t);
                }
                cout << "ranking done!\n";
                cout << t << endl << rates.size() << endl;

//                std::sort(rates.begin(), rates.end(), n_sort);
                for (int k = 0; k < t; ++k) {
                    ptchs[k].rate = rates[k].r;

                    if (debug) {
                        // cout << ptchs[k].no << " " << rates[k].no << " " << rates[k].r << endl;
                        imwrite("bmp/" + to_string(i) + "-" + to_string(j) + "/patch" + to_string(i) + "-" +
                                to_string(j) + "pos" + to_string(k) + "s_" + to_string(ptchs[k].MSSIM)+ "r_" +
                                to_string(rates[k].r) + ".png", rois[k]);
                    }
                }
//                std::sort(ptchs.begin(), ptchs.end(), util::ss_sort_dec);
//                ptchs.pop_back();
                std::sort(ptchs.begin(), ptchs.end(), util::p_sort_dec);
                int c = ptchs.size() / 3;
                if (c == 0 && ptchs.size() > 1)
                    c++;
                vector<patch> chunk(ptchs.begin(), ptchs.begin() + c);
                vector<patch>/* selchunk,*/ssimsorted(ptchs.begin(), ptchs.begin() + c);
                patch pp = ptchs[0];
//                for (std::vector<int>::size_type i = 1; i != ptchs.size(); i++) {
//                    if (getMSSIM(rois[pp.no], rois[ptchs[i].no]) < 0.8) {
//                        pp = ptchs[i];
//                    } else
//                        break;
//                }
                std::vector<cv::Mat> ptch_mat;
//                for (auto &ch : chunk) {
//                    ptch_mat.push_back(rois[ch.no]);
//                }
                Mat sel_p;
//                sel_p = util::getMean(ptch_mat);
//                for (auto &ch : chunk) {
//                    if (util::getMSSIM(rois[ch.no], sel_p) > 0.5)
//                        selchunk.push_back(ch);
//                }
                ptch_mat.clear();
                for (auto &ch :  chunk) {
                    ptch_mat.push_back(rois[ch.no]);
                }
                if (chunk.size() > 1/* && false*/) {
//                    std::sort(selchunk.begin(), selchunk.end(), util::p_sort_dec);
                    pp = chunk[0];
                    //check ssim
                    std::sort(ssimsorted.begin(), ssimsorted.end(), util::p_ssim_sort_dec);
                    if (pp.rate - ssimsorted[0].rate < 0.5)
                        pp = ssimsorted[0];
                    sel_p = rois[pp.no];
//TODO
//                    if (pp.MSSIM < 0.7) {
//                        Mat H = FindHomoRobust(ref, sel_p, Mat());
//                        if (!H.empty()) {
//                            Mat wimg2, gw;
//                            warpPerspective(sel_p, wimg2, H, ref.size(), 2);
//                            cv::cvtColor(wimg2, gw, COLOR_BGR2GRAY);
//
//                            double ssim = util::getMSSIM(gw, ref);
//                            if (ssim > pp.MSSIM)
//                                sel_p = wimg2;
//                        }
//                    }
//                    cout << "selected patch:" << i << "-" << j << "=frame:" << selchunk[0].no << endl;
//                    sel_p = util::getMean(ptch_mat);

//                    Mat image;
//                    GaussianBlur(sel_p, image, cv::Size(5, 5), 3);
//                    addWeighted(sel_p, 1.5, image, -0.5, 0, sel_p);

                } else {
                    sel_p = rois[pp.no];
                    cout << "selected patch:" << i << "-" << j << "=frame:" << pp.no << endl;
                }
                Mat mask(img1_clone.size(), CV_8UC1, Scalar::all(0));
                Mat froi(img1_clone.size(), CV_8UC3, Scalar::all(0));
                Rect r(pp.x + 4, pp.y + 4, wsize.width - 8, wsize.height - 8);
                cv::Mat pRoi = mask(r);
                pRoi.setTo(cv::Scalar(255));
                cv::imwrite("pRoi.png", mask);
                cv::Mat mRoi = froi(Rect(pp.x, pp.y, wsize.width, wsize.height));
//                    froi(r) = rois[pp.no];
                sel_p.copyTo(mRoi);
//                cv::imwrite(to_string(pp.no) + " " + to_string(pp.x) + " " + to_string(pp.y) + "froi.png", froi);
//                    cv::imwrite("rois.png", rois[pp.no]);
                cout << "blending " + to_string(pp.no) + " " + to_string(pp.x) + " " + to_string(pp.y) << endl;
                if (debug) {
                    imwrite("bmp/" + to_string(i) + "-" + to_string(j) + "/" + to_string(pp.no) + "patch" + ".png",
                            sel_p);
                }
                blend::seamlessBlend(froi, img1_clone, mask, img1_clone);
            }
            rates.clear();
            rates.shrink_to_fit();
            ptchs.clear();
            ptchs.shrink_to_fit();
            j += oy / stride;
            if (j > img1_clone.rows - wsize.height && tj) {
                tj = false;
                j = min(j, img1_clone.rows - wsize.height) - 2;
            }
        }
        i += ox / stride;
        if (i > img1_clone.cols - wsize.width && ti) {
            ti = false;
            i = min(i, img1_clone.cols - wsize.width) - 2;
        }
    }
    cv::imwrite("seamless-blend.png", img1_clone);

    return img1_clone;
}

int main(int argc, char *argv[]) {
//    Mat sa = imread("sc1.png", 0), sb = imread("sc2.png", 0),
//            sc = imread("sa3.png", 0);
//
//    double ssim = util::getMSSIM(sb, sa);
//    cout << ssim << endl;
//    return 0;
//
//    Rect r(700, 900, 300, 300);
//    Mat roi = sa(r);
//
//    Mat rroi = sb(r);
//
//      ssim = util::getMSSIM(rroi, roi);
//    cout << ssim << endl;
//    rroi = sc(r);
//    ssim = util::getMSSIM(roi, rroi);
//    cout << ssim << endl;
//    return 0;
    string deb = "debug";
    for (int i = 0; i < argc; ++i) {
        if (deb.compare(argv[i]) == 0)
            debug = true;
    }
    mclmcrInitialize();

    fishInitialize();

    VideoCapture capture("input.mp4");

    vector<Point2f> p = util::read_points("task_data.json");
    Point2f s_out = util::read_out_size("task_data.json");
    int fr = util::read_ref_frame("task_data.json");
    capture.set(CV_CAP_PROP_POS_FRAMES, fr);
    capture >> image_ref;
    vector<Point2f> dst;
    image_nowarp = image_ref.clone();

    dst.push_back(Point2f(0, 0));
    dst.push_back(Point2f(s_out.x, 0));
    dst.push_back(Point2f(s_out.x, s_out.y));
    dst.push_back(Point2f(0, s_out.y));

    Mat HOMO = util::four_point_transform_homo(image_ref, p, dst);
    image_ref = util::four_point_transform(image_ref, p, dst);

    if (capture.isOpened()) {
        process(capture, image_ref, HOMO);
    } else {
        cout << "not found!" << endl;
    }

    return 0;
}


Mat
FindHomoRobust(Mat ref, Mat frame, Mat ref_mask) {

    RMatcher rMatcher;
    vector<KeyPoint> kpts1;
    Mat desc1;
    cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(/*nfeatures, nOctaveLayers, contrastThreshold,
                                                                edgeThreshold, sigma*/);

    sift->detect(ref, kpts1, ref_mask);
    sift->compute(ref, kpts1, desc1);


    int ry = frame.rows, cx = frame.cols, split, xoy;
    xoy = ry > cx ? 0 : 1;
    split = xoy ? cx / 2 : ry / 2;


//    cvtColor(frame, frame, CV_BGR2GRAY);
    vector<KeyPoint> kpts2;
    Mat desc2;
    sift->detect(frame, kpts2);
    sift->compute(frame, kpts2, desc2);

    std::vector<cv::DMatch> matches;
    bool bSymmetricMatches = true;//caution, activate this with knn matching will cause errors.
    cv::BFMatcher matcher(cv::NORM_L2, bSymmetricMatches);
    if (bSymmetricMatches) {
        matcher.match(desc1, desc2, matches);
    }

    Mat homography;

    std::vector<DMatch> good_matches;


    run_kvld(ref, frame, matches, kpts2, kpts1, good_matches);


    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (int i = 0; i < good_matches.size(); i++) {
        //-- Get the keypoints from the good matches
        obj.push_back(kpts2[good_matches[i].queryIdx].pt);
        scene.push_back(kpts1[good_matches[i].trainIdx].pt);
    }

    if (good_matches.size() > 5) {
        homography = findHomography(obj, scene, RANSAC);
    }

    return homography;
}

double rank_blur_fish(Mat img, int i) {
    string str = "bmp/p" + to_string(i) + ".png";
    imwrite(str, img);
    mwArray path = mwArray(str.c_str());
    mwArray res;
    fish(1, res, path);

    double s = res;

    // cout << i << "  :  " << s << endl;
    frame f;
    f.r = s;
    f.no = i;
    rates.push_back(f);
    // imwrite("bmp/"+to_string(i)+"frame:"+to_string(s)+".jpg",img);
    return s;
}