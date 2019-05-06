#include <iostream>
#include <opencv2/opencv.hpp>
#include "Dewarper.cpp"
#include "Dewarper_CPU.h"
#include "json.hpp"
#include "Scaler.cpp"
//#include "Assesors/VOL.cpp"
#include "Assesors/rank_bluro.h"
#include "Welder/Composer.h"
//#include "blend/blend.h"

using json = nlohmann::json;

using namespace std;
using namespace cv;

int parseOptionLoc(int argc, char** argv, const char* option){
    for (int i =1 ; i < argc; i++){
        if (strcmp(argv[i],option) == 0)
            return i+1;
    }

    return -1;
}


const char *usage =
        "USAGE: VideoDocRec -v  <input_video_path> -d  <input_video_data_json_path>  -o <output_image_path> "
        "[-q  fish/bluro/vol] [--gpu true]\n\n"

        "Options:"
        "\n  -v  path to input video file"
        "\n  -d  path to data json file which contain video details"
        "\n  -o  output image destination"
        "\n  -q specify quality algorithm to use. (Default : vol)"
        "\n  --gpu if want to run program on gpu. (The default is cpu only)"
        "\nExamples:\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q fish\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png --gpu true\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q vol --gpu true\n";

const char *gpuUsage =
        "gpu option error\nIf you want to use gpu to run program pass \"--gpu true\" option to program\n\n"
        "USAGE: VideoDocRec -v  <input_video_path> -d  <input_video_data_json_path>  -o <output_image_path> "
        "[-q  fish/bluro/vol] [--gpu true]\n\n"

        "Options:"
        "\n  -v  path to input video file"
        "\n  -d  path to data json file which contain video details"
        "\n  -o  output image destination"
        "\n  -q specify quality algorithm to use. (Default : vol)"
        "\n  --gpu if want to run program on gpu. (The default is cpu only)"
        "\nExamples:\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q fish\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png --gpu true\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q vol --gpu true\n";

const char *qualityUsage =
        "quality option error\nIf you want to use gpu to run program pass \"-q fish\" or \"-q bluro\" or \"-q vol\" option to program\n\n"
        "USAGE: VideoDocRec -v  <input_video_path> -d  <input_video_data_json_path>  -o <output_image_path> "
        "[-q  fish/bluro/vol] [--gpu true]\n\n"

        "Options:"
        "\n  -v  path to input video file"
        "\n  -d  path to data json file which contain video details"
        "\n  -o  output image destination"
        "\n  -q specify quality algorithm to use. (Default : vol)"
        "\n  --gpu if want to run program on gpu. (The default is cpu only)"
        "\nExamples:\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q fish\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png --gpu true\n"
        "  VideoDocRec -v input.mp4 -d data.json -o output.png -q vol --gpu true\n";

Mat blendImage(Mat,Mat, int,int);


int main(int argc, char** argv) {

    // Program arguments
    char *videoPath;
    char *outputPath;
    char *dataPath;
    char *quality;
    bool isGpu = false;


    if ( parseOptionLoc(argc,argv,"-v") == -1 ||  parseOptionLoc(argc,argv,"-d") == -1
    ||  parseOptionLoc(argc,argv,"-o") == -1 || argc > 11)
    {
        cout << usage;
        return 0;
    } else{
        videoPath = argv[parseOptionLoc(argc,argv,"-v")];
        outputPath = argv[parseOptionLoc(argc,argv,"-o")];
        dataPath = argv[parseOptionLoc(argc,argv,"-d")];
    }


    int gLoc = parseOptionLoc(argc,argv,"--gpu");
    if ( gLoc != -1 )
    {
        char *option = argv[gLoc];
        if (String(option) != "true") {
            cout << gpuUsage;
            return 0;
        } else{
            isGpu = true;
        }
    }

    int qLoc = parseOptionLoc(argc,argv,"-q");
    if ( qLoc != -1 )
    {
        char *option = argv[qLoc];
        if (String(option) != "fish" && String(option) != "bluro" && String(option) != "vol") {
            cout << qualityUsage;
            return 0;
        } else{
            quality = option;
        }
        if (String(option) == "fish"){
            // Initialize matlab
            mclmcrInitialize();
            fishInitialize();
        }
    }




    cout <<dataPath<<endl;
    // Important fields
    Mat featureMask;
    Mat refFrame;
    Mat dewarpedRefFrame;
    Mat scores;
    Mat scoredImages;
    Mat dewarpMask;


    // Reading Video
    VideoCapture capture = VideoCapture(videoPath);

    // Finding ref frame
    ifstream i(dataPath);
    json data;
    i>> data;
    int refFrameIndex = data["reference_frame_id"];

    int numFrames;
    for(numFrames = 0; capture.isOpened(); numFrames++){

        Mat image;
        bool isRead = capture.read(image);
        if(isRead) {
            if ( numFrames == refFrameIndex){
                refFrame = image;
                //cvtColor(refFrame, refFrame, COLOR_RGB2GRAY);
            }
        } else
            break;
    }

    // Compute srcHomography (the matrix which transform ref image to target dimensions)
    // and targetDimen ( target dimension)
    Scaler scaler = Scaler(dataPath);
    Mat srcHomography;
    Size targetDimen;
    scaler.getHomoAndDimen(srcHomography, targetDimen);

    warpPerspective(refFrame, dewarpedRefFrame,srcHomography,targetDimen);


    // Reading video and process frames
    capture = VideoCapture(videoPath);
    featureMask = (Mat::ones(refFrame.size(),CV_8U))*255;
    scores = Mat::zeros(targetDimen,CV_64FC3);
    scoredImages = Mat::zeros(targetDimen,CV_64FC3);

    // Initialize composer
    Composer_old composer(256,dewarpedRefFrame,scoredImages,scores, quality, isGpu);

    for (int i =0; capture.isOpened(); i++){


        Mat inProcImage;
        bool isRead = capture.read(inProcImage);

        if (isRead){ {
                cout << "frame : " << i << endl;

                Mat dewarpedImage;

                if (i == refFrameIndex) {
                    dewarpedImage = dewarpedRefFrame;
                    featureMask = (Mat::ones(refFrame.size(),CV_8U))*255;
                    dewarpMask = (Mat::ones(dewarpedRefFrame.size(),CV_8U))*255;
                    cout << "ref Frame used";
                } else {
                    if(isGpu) {
                        dewarpedImage = Dewarper::warp(refFrame, inProcImage, featureMask, srcHomography, targetDimen,
                                                       dewarpMask);

                    } else{
                        dewarpedImage = Dewarper_CPU::warp(refFrame, inProcImage, featureMask, srcHomography, targetDimen
                                ,dewarpMask); //TODO it seems that clear warp is not working


//                        std::ostringstream out1;
//                        out1 << "../output/" << i <<"_orig.png";
//                        string outputFilePath = out1.str();
//                        imwrite(outputFilePath,inProcImage);
//
//                        std::ostringstream out2;
//                        out2 << "../output/" << i <<".jpg";
//                        string outputFilePath2 = out2.str();
//                        imwrite(outputFilePath2,dewarpedImage);
                    }

                }
                if (dewarpedImage.rows == 0 || dewarpedImage.cols == 0)
                    continue;

                composer.compose(dewarpedImage,dewarpMask,i,1);


            }
        } else
            break;


    }


    //region Test warp function
    scores = composer.getScores();
    scoredImages = composer.getScoredImages();

    Mat cImg[3];
    Mat cScr[3];

    split(scoredImages,cImg);
    split(scores,cScr);

    cImg[0] /= cScr[0];
    cImg[1] /= cScr[1];
    cImg[2] /= cScr[2];
    Mat finalImage;
    merge(cImg,3,finalImage);
    finalImage.convertTo(finalImage, CV_8UC3);

    //TODO put all program in try and exception and in all situations matlab should be terminated

//     finalImage = blendImage(finalImage, dewarpedRefFrame,256,128);

    imwrite(outputPath,finalImage);


    return 0;
}

//Mat blendImage(Mat finalImage,Mat refImage,int windowSize,int step) {
//    Mat blendedImage = refImage.clone();
//
//    for (int i = 0; i < finalImage.cols; i += step) {
//        for (int j = 0; j < finalImage.rows; j += step) {
//            Rect roi,mask_rect;
//            int OWidth = finalImage.cols - i;
//            int OHeight = finalImage.rows - j;
//
//            if (i + windowSize > finalImage.cols && j + windowSize > finalImage.rows)
//                roi = Rect(i, j, OWidth, OHeight);
//            else if (i + windowSize > finalImage.cols)
//                roi = Rect(i, j, OWidth, windowSize);
//            else if (j + windowSize > finalImage.rows)
//                roi = Rect(i, j, windowSize, OHeight);
//            else
//                roi = Rect(i, j, windowSize, windowSize);
//
//
//            Mat mask(finalImage.size(), CV_8UC1, Scalar::all(0));
//            Mat froi(finalImage.size(), CV_8UC3, Scalar::all(0));
//
//            mask_rect=Rect(roi.x+4,roi.y+4,roi.width-8,roi.height-8);
//            if (mask_rect.width <= 0 || mask_rect.height <= 0 )
//                continue;
//
//            mask(mask_rect).setTo(cv::Scalar(255));
//
//
//            Mat win = finalImage(roi);
//
//            Mat mroi =froi(roi);
//            Mat myroi=win.clone();
//            myroi.copyTo(mroi);
//
//
//            blend::seamlessBlend(froi,blendedImage,mask,blendedImage);
//        }
//
//    }
//
//    return blendedImage;
//}