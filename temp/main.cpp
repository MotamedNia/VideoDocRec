#include "util.h"

int main(int argc, char *argv[]) {
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
}

