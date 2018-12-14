# SmartDoc

### Introduction
This paper addresses the problem of reconstructing a high-resolution image from 
a video sequence. The application of this method is useful when we want to scan 
a document with a mobile device. In this scenario, one can record a video of a 
document and capture close shots of that document so the details appear on the video. 
By having such a video, our method first align video frames to one viewpoint and 
then stitch them to the final image. In this work, We have extended the accuracy 
of alignment with eliminating keypoints search space. Also, a sharpness-aware 
approach is presented for image stitching. We evaluated our method on SmartDoc 
Video dataset and reported the results.

### Requirements

* [CMake](https://cmake.org/) : To generate  build files
* [OpenCV](https://opencv.org/) : To perform Computer vision functions.
* [Json](https://github.com/nlohmann/json) : To parse data.json file
    * Put json.hpp to the project root directory.

#### Optianal requirements
* [FISH  Fast Wavelet-Based Algorithm for Global and Local Image Sharpness Estimation](https://sites.google.com/site/vapovu/papers) 
: For quality assessing related functions.
    * Download matlab code and create c++ library. The you can follow this [gist.github](https://gist.github.com/minooei/1ec439dd91857d35d7b2d963056f2a45)
    to enable the project to use this library.     
### Building
Simply create build directory and run these commands:

```shell
cmake -DCMAKE_BUILD_TYPE=Release
make
```

## Running
To run project , run such command inside build directory
```shell
VideoDocRec -v  <input_video_path> -d  <input_video_data_json_path>  -o <output_image_path>
```
These are some particular examples :
```shell
VideoDocRec -v input.mp4 -d data.json
VideoDocRec -v input.mp4 -d data.json -q fish
VideoDocRec -v input.mp4 -d data.json --gpu true
VideoDocRec -v input.mp4 -d data.json -q vol -gpu true
```


## License

