#include "shot_trace.h"

#include "opencv2/highgui.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>


using namespace cv::dnn;

void testing_detect_ball_PC_IOS()
{
    bool frame_from_PC = false;
    string mask_frame_folder;
    string frame_folder;
    string check_output_folder;
    if (frame_from_PC)
    {
        mask_frame_folder = "frame_from_PC/saved_masks";
        frame_folder = "frame_from_PC/saved_outputs";
        check_output_folder = "./debug_outputs_PC";
    }
    else
    {
        mask_frame_folder = "frame_from_IOS/saved_masks";
        frame_folder = "frame_from_IOS/saved_outputs";
        check_output_folder = "./debug_outputs_IOS";
    }

    utils::fs::remove_all(check_output_folder);
    utils::fs::createDirectory(check_output_folder);

    Ptr<SimpleBlobDetector> golfball_detector = st::create_ball_detector(0.1, 0.1, 0.01);
    std::vector<float> the_first_golfball_point{ 782, 1149 };
    std::vector<float> the_best_reference_point{ 746.67, 1000.18 };
    std::vector<float> vect = { the_first_golfball_point[0] - the_best_reference_point[0],
                                    the_first_golfball_point[1] - the_best_reference_point[1] };
    float std_angle = std::atan2(vect[0], vect[1]);
    std_angle = std_angle * (180.0 / st::PI_F);
    cout << "check std_angle: " << std_angle << endl;
    float max_dis = std::sqrt(vect[0] * vect[0] + vect[1] * vect[1]);
    cout << "check max_dis: " << max_dis << endl;
    std::vector<float> previous_center_point = the_first_golfball_point;
    int count_ball = 0;
    int count_undetect = 0;
    int count_undetect_max = 3;
    float max_tolerance_angle = 2;
    float mean_ball_radius = 0;
    std::vector<float> list_ball_radius;
    std::vector<float> list_sequence_dis;

    for (int frame_num = 63; frame_num < 150; ++frame_num)
    {
        string file = to_string(frame_num) + ".png";
        string mask_frame_path = utils::fs::join(mask_frame_folder, file);
        //cout << mask_frame_path << endl;
        Mat mask_frame = cv::imread(mask_frame_path, 0);
        string frame_path = utils::fs::join(frame_folder, file);
        //cout << mask_frame_path << endl;
        Mat frame = cv::imread(frame_path, 1);
        std::vector<KeyPoint> keypoints;
        golfball_detector->detect(mask_frame, keypoints);
        cout << "keypoints size: " << keypoints.size() << endl;
        std::vector<float> selected_center_point = previous_center_point;
        float selected_rad = 0;
        if (keypoints.size() != 0)
        {
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                float rad = keypoints[i].size / 2;
                Point2f center_point = keypoints[i].pt;
                if (center_point.y > the_first_golfball_point[1] - 5)
                {
                    cout << "Removed by the_first_golfball_point" << endl;
                    continue;
                }
                std::vector<float> vect = { the_first_golfball_point[0] - center_point.x,
                                            the_first_golfball_point[1] - center_point.y };
                float angle = std::atan2(vect[0], vect[1]);
                angle = angle * (180.0 / st::PI_F);
                if (abs(angle - std_angle) > max_tolerance_angle)
                {
                    cout << "Removed by std_angle" << endl;
                    continue;
                }
                float dis = st::distance(std::vector<float>{ center_point.x, center_point.y },
                    previous_center_point);
                if (dis > max_dis)
                {
                    cout << "Removed by max_dis" << endl;
                    continue;
                }
                if (list_sequence_dis.size() > 5)
                {
                    std::vector<float> cut_list_sequence_dis(list_sequence_dis.end() - 5, list_sequence_dis.end());
                    float median_dis = st::np_median(cut_list_sequence_dis);
                    if (dis > 5 * median_dis)
                    {
                        cout << "Removed by sequence_dis " << endl;
                        continue;
                    }
                }
                if (mean_ball_radius != 0 && rad > 2 * mean_ball_radius)
                {
                    cout << "Removed by mean_ball_radius " << endl;
                    continue;
                }
                if (center_point.y >= previous_center_point[1])
                {
                    cout << "Removed by greater previous_center_point" << endl;
                    continue;
                }
                if (selected_rad < rad)
                {
                    selected_rad = rad;
                    selected_center_point = std::vector<float>{ center_point.x, center_point.y };
                }
            }
        }
        if (selected_rad != 0)
        {
            float dis = st::distance(selected_center_point, previous_center_point);
            list_sequence_dis.push_back(dis);
            list_ball_radius.push_back(selected_rad);
            count_ball += 1;
            count_undetect = 0;
            previous_center_point = selected_center_point;
            cv::circle(frame, Point(selected_center_point[0], selected_center_point[1]),
                int(selected_rad), Scalar(255, 0, 0), 2);
        }
        else
        {
            count_undetect += 1;
            selected_center_point = previous_center_point;
            std::string text = "Undetected " + std::to_string(frame_num);
            cv::putText(frame, text, cv::Point(50, 200), cv::FONT_HERSHEY_PLAIN,
                5, Scalar(0, 0, 255), 5, cv::LINE_AA);
        }
        if (list_ball_radius.size() >= 3)
        {
            std::vector<float> cut_list_ball_radius(list_ball_radius.end() - 3, list_ball_radius.end());
            mean_ball_radius = st::mean_1D(cut_list_ball_radius);
        }
        string save_path = utils::fs::join(check_output_folder, file);
        imwrite(save_path, frame);
        if (count_undetect == 3)
        {
            cout << "No need detect any more" << endl;
            break;
        }
    }
}


void creating_masks(string video_path, int IM_frame)
{
    //Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorKNN(3, 300, true);
    Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
    string check_output_folder = "./saved_frames";
    string check_mask_folder = "./saved_masks";
    utils::fs::remove_all(check_output_folder);
    utils::fs::createDirectory(check_output_folder);
    utils::fs::remove_all(check_mask_folder);
    utils::fs::createDirectory(check_mask_folder);
    Mat frame, mask_frame;
    VideoCapture cap(video_path);
    int frame_num = 0;
    while (cap.read(frame))
    {
        if (frame_num >= IM_frame - 10)
        {
            backSub->apply(frame, mask_frame);
            mask_frame.setTo(0, mask_frame == 127);
        }
        if (frame_num >= IM_frame)
        {
            string file = to_string(frame_num) + ".png";
            string save_path = utils::fs::join(check_mask_folder, file);
            imwrite(save_path, mask_frame);
            save_path = utils::fs::join(check_output_folder, file);
            imwrite(save_path, frame);
        }
        frame_num++;
    }
}


void testing_detecting_golfball()
{
    std::vector<std::vector<float>> list_golfball_center_points;
    string video_path = "testing_samples/20240320_038_cut.mp4";
    std::vector<float> the_first_golfball_point{ 782, 1149 };
    std::vector<float> the_best_reference_point{ 746.67, 1000.18 };
    int IM_frame = 63;
    st::detecting_golfball(video_path,
        list_golfball_center_points,
        the_first_golfball_point,
        the_best_reference_point,
        IM_frame, true);
    cout << "list_golfball_center_points size: " << list_golfball_center_points.size() << endl;
}


void testing_detect_ball_singleframe(string frame_path, string mask_frame_path,
    std::vector<float> the_first_golfball_point,
    std::vector<float> the_best_reference_point)
{
    string check_output_folder = "debug_outputs";
    utils::fs::remove_all(check_output_folder);
    utils::fs::createDirectory(check_output_folder);

    Ptr<SimpleBlobDetector> golfball_detector = st::create_ball_detector(0.1, 0.1, 0.01);
    std::vector<float> vect = { the_first_golfball_point[0] - the_best_reference_point[0],
                                    the_first_golfball_point[1] - the_best_reference_point[1] };
    float std_angle = std::atan2(vect[0], vect[1]);
    std_angle = std_angle * (180.0 / st::PI_F);
    cout << "check std_angle: " << std_angle << endl;
    float max_dis = std::sqrt(vect[0] * vect[0] + vect[1] * vect[1]);
    cout << "check max_dis: " << max_dis << endl;
    std::vector<float> previous_center_point = the_first_golfball_point;
    float max_tolerance_angle = 2;
    Mat mask_frame = cv::imread(mask_frame_path, 0);
    Mat frame = cv::imread(frame_path, 1);

    std::vector<KeyPoint> keypoints;
    golfball_detector->detect(mask_frame, keypoints);
    cout << "keypoints size: " << keypoints.size() << endl;
    std::vector<float> selected_center_point = previous_center_point;
    float selected_rad = 0;
    if (keypoints.size() != 0)
    {
        for (size_t i = 0; i < keypoints.size(); i++)
        {
            float rad = keypoints[i].size / 2;
            Point2f center_point = keypoints[i].pt;
            //cv::circle(frame, Point(center_point.x, center_point.y),
                //int(rad), Scalar(255, 0, 0), 2);
            if (center_point.y > the_first_golfball_point[1] - 5)
            {
                cout << "Removed by the_first_golfball_point" << endl;
                //cv::circle(frame, Point(center_point.x, center_point.y),
                //int(rad), Scalar(255, 0, 0), 2);
                continue;
            }
            std::vector<float> vect = { the_first_golfball_point[0] - center_point.x,
                                        the_first_golfball_point[1] - center_point.y };
            float angle = std::atan2(vect[0], vect[1]);
            angle = angle * (180.0 / st::PI_F);
            if (abs(angle - std_angle) > max_tolerance_angle)
            {
                cout << "Removed by std_angle" << endl;
                //cv::circle(frame, Point(center_point.x, center_point.y),
                    //int(rad), Scalar(255, 0, 0), 2);
                continue;
            }
            float dis = st::distance(std::vector<float>{ center_point.x, center_point.y },
                previous_center_point);
            if (dis > max_dis)
            {
                cout << "Removed by max_dis: "  << dis << endl;
                //cv::circle(frame, Point(center_point.x, center_point.y),
                    //int(rad), Scalar(255, 0, 0), 2);
                continue;
            }
            if (selected_rad < rad)
            {
                selected_rad = rad;
                selected_center_point = std::vector<float>{ center_point.x, center_point.y };
            }
            
        }
    }
    if (selected_rad != 0)
    {
        cv::circle(frame, Point(selected_center_point[0], selected_center_point[1]),
            int(selected_rad), Scalar(0, 255, 0), 2);
    }
    else
        cout << "Detect failed" << endl;//*/
    string save_path = utils::fs::join(check_output_folder, "check.png");
    imwrite(save_path, frame);
}


void draw_ball_trace(Mat& image, std::vector<std::vector<float>> list_golfball_center_points)
{
    for (size_t i = 0; i < list_golfball_center_points.size() - 1; ++i)
    {
        cv::Point points1(list_golfball_center_points[i][0], list_golfball_center_points[i][1]);
        cv::Point points2(list_golfball_center_points[i + 1][0], list_golfball_center_points[i + 1][1]);
        cv::line(image, points1, points2, cv::Scalar(0, 0, 255), 7);
    }
}


void testing_detect_ball_from_IOS(string mask_frame_folder,
    string frame_folder, 
    int IM_frame, int End_frame,
    std::vector<float> the_first_golfball_point,
    std::vector<float> the_best_reference_point)
{
    string check_output_folder = "./debug_outputs";
    utils::fs::remove_all(check_output_folder);
    utils::fs::createDirectory(check_output_folder);

    Ptr<SimpleBlobDetector> golfball_detector = st::create_ball_detector(0.1, 0.1, 0.01);
    std::vector<float> vect = { the_first_golfball_point[0] - the_best_reference_point[0],
                                    the_first_golfball_point[1] - the_best_reference_point[1] };
    float std_angle = std::atan2(vect[0], vect[1]);
    std_angle = std_angle * (180.0 / st::PI_F);
    cout << "check std_angle: " << std_angle << endl;
    float max_dis = std::sqrt(vect[0] * vect[0] + vect[1] * vect[1]);
    cout << "check max_dis: " << max_dis << endl;
    std::vector<float> previous_center_point = the_first_golfball_point;
    int count_ball = 0;
    int count_undetect = 0;
    int count_undetect_max = 3;
    float max_tolerance_angle = 2;
    float mean_ball_radius = 0;
    std::vector<float> list_ball_radius;
    std::vector<float> list_sequence_dis;
    std::vector<std::vector<float>> list_golfball_center_points;
    Mat mask_frame, frame;
    for (int frame_num = IM_frame; frame_num < End_frame+1; ++frame_num)
    {
        string file = to_string(frame_num) + ".png";
        string mask_frame_path = utils::fs::join(mask_frame_folder, file);
        //cout << mask_frame_path << endl;
        mask_frame = cv::imread(mask_frame_path, 0);
        string frame_path = utils::fs::join(frame_folder, file);
        //cout << mask_frame_path << endl;
        frame = cv::imread(frame_path, 1);
        std::vector<KeyPoint> keypoints;
        golfball_detector->detect(mask_frame, keypoints);
        cout << "keypoints size: " << keypoints.size() << endl;
        std::vector<float> selected_center_point = previous_center_point;
        float selected_rad = 0;
        if (keypoints.size() != 0)
        {
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                float rad = keypoints[i].size / 2;
                Point2f center_point = keypoints[i].pt;
                if (center_point.y > the_first_golfball_point[1] - 5)
                {
                    cout << "Removed by the_first_golfball_point" << endl;
                    continue;
                }
                std::vector<float> vect = { the_first_golfball_point[0] - center_point.x,
                                            the_first_golfball_point[1] - center_point.y };
                float angle = std::atan2(vect[0], vect[1]);
                angle = angle * (180.0 / st::PI_F);
                if (abs(angle - std_angle) > max_tolerance_angle)
                {
                    cout << "Removed by std_angle" << endl;
                    continue;
                }
                float dis = st::distance(std::vector<float>{ center_point.x, center_point.y },
                    previous_center_point);
                if (dis > max_dis+5)
                {
                    cout << "Removed by max_dis" << endl;
                    continue;
                }
                if (list_sequence_dis.size() > 5)
                {
                    std::vector<float> cut_list_sequence_dis(list_sequence_dis.end() - 5, list_sequence_dis.end());
                    float median_dis = st::np_median(cut_list_sequence_dis);
                    if (dis > 5 * median_dis)
                    {
                        cout << "Removed by sequence_dis " << endl;
                        continue;
                    }
                }
                if (mean_ball_radius != 0 && rad > 2 * mean_ball_radius)
                {
                    cout << "Removed by mean_ball_radius " << endl;
                    continue;
                }
                if (center_point.y >= previous_center_point[1])
                {
                    cout << "Removed by greater previous_center_point" << endl;
                    continue;
                }
                if (selected_rad < rad)
                {
                    selected_rad = rad;
                    selected_center_point = std::vector<float>{ center_point.x, center_point.y };
                }
            }
        }
        if (selected_rad != 0)
        {
            float dis = st::distance(selected_center_point, previous_center_point);
            list_sequence_dis.push_back(dis);
            list_ball_radius.push_back(selected_rad);
            count_ball += 1;
            count_undetect = 0;
            previous_center_point = selected_center_point;
            cv::circle(frame, Point(selected_center_point[0], selected_center_point[1]),
                int(selected_rad), Scalar(255, 0, 0), 2);
        }
        else
        {
            count_undetect += 1;
            selected_center_point = previous_center_point;
            std::string text = "Undetected " + std::to_string(frame_num);
            cv::putText(frame, text, cv::Point(50, 200), cv::FONT_HERSHEY_PLAIN,
                5, Scalar(0, 0, 255), 5, cv::LINE_AA);
        }
        if (list_ball_radius.size() >= 3)
        {
            std::vector<float> cut_list_ball_radius(list_ball_radius.end() - 3, list_ball_radius.end());
            mean_ball_radius = st::mean_1D(cut_list_ball_radius);
        }
        string save_path = utils::fs::join(check_output_folder, file);
        cv::imwrite(save_path, frame);
        if (count_undetect == 3)
        {
            cout << "No need detect any more" << endl;
            break;
        }
        list_golfball_center_points.push_back(selected_center_point);
    }
    //Draw ball trace
    draw_ball_trace(frame, list_golfball_center_points);
    string ball_trace_image_path = utils::fs::join(check_output_folder, "ball_trace_image.png");
    cv::imwrite(ball_trace_image_path, frame);
}


int main00()
{
    // Detect from IOS
    string frame_folder = "debug_20240320_008_from_IOS/saved_frames";
    string mask_frame_folder = "debug_20240320_008_from_IOS/saved_masks";
    std::vector<float> the_first_golfball_point{ 857, 1045 };
    std::vector<float> the_best_reference_point{ 764.48, 773.194 };
    testing_detect_ball_from_IOS(mask_frame_folder,
        frame_folder,
        175, 309,
        the_first_golfball_point,
        the_best_reference_point);
    return 0;
}


int main11()
{
    // Loading AI model
    string model_path = "model/test_model.onnx";
    Net test_model = readNetFromONNX(model_path);

    return 0;
}


int main22()
{
    string video_path = "testing_samples/20240320_008.mp4";
    string trajectory_data_path = "trajectory_sample/full_trajectory_data.txt";
    st::draw_ball_trajectory(video_path,
        trajectory_data_path,
        174, 10, 24);
    return 0;
}


int mainm() //full testing
{
    // Loading AI model
    string model_path = "model/detecting_header_model.onnx";
    st::loading_model(model_path);
    // Getting the_first_golfball_point 
    string video_path = "testing_samples/20240320_038_cut.mp4"; 
    //20240320_038_cut GolfswingHD.mp4 20240320_008 20240320_017.mov
    //std::vector<float> the_first_golfball_point{ 782, 1149 }; //777,1151 782, 1149
    //string video_path = "testing_samples/indoor.mp4";
    std::vector<float> the_first_golfball_point;
    st::setting_first_position_of_golfball(video_path,
        the_first_golfball_point,
        24, 0.5);
    cout << "the_first_golfball_point: " << the_first_golfball_point[0] << ", " << the_first_golfball_point[1] << endl;
    //*/
    // Run Shot trace process
    string shot_trace_output_video_path = "shot_trace_output.mp4";
    int write_fps = 10;
    int debug_header = false;
    int debug_ball = true;
    string trace_image_path = "trace_image.png";
    st::creating_shot_trace(video_path,
        shot_trace_output_video_path,
        trace_image_path,
        the_first_golfball_point,
        write_fps, debug_header, debug_ball);

    return 0;
}


int mainxx() //Test Detect Address
{
    // Loading AI model
    string model_path = "model/detecting_header_model.onnx";
    st::loading_model(model_path);
    string video_path = "testing_samples/20240320_038_cut.mp4";
    
    std::vector<std::vector<float>> list_header_center_points = {
    {778.65, 1160.1},
    {778.65, 1160.1},
    {778.73, 1160.2},
    {778.89, 1160.3},
    {779.77, 1160.2},
    {781.05, 1160.4},
    {781.52, 1160.5},
    {780.68, 1160.7},
    {781.84, 1160.6},
    {782.08, 1160.5},
    {783.91, 1161},
    {783.86, 1161},
    {788.1, 1165.9},
    {788.33, 1166},
    {792.62, 1175.6},
    {792.5, 1175.8},
    {796.83, 1186.3},
    {796.72, 1186.3},
    {796.25, 1189},
    {796.18, 1189.1},
    {776.72, 1185.2},
    {776.57, 1185.3},
    {731.37, 1148.1},
    {695.14, 1113.4},
    {648.79, 1066},
    {596.82, 1004.8},
    {532.3, 931.76},
    {474.1, 841.03},
    {404.74, 750.68}
    };
    int AD_frame;
    std::vector<int> AD_coord;
    std::vector<std::vector<float>> sliced_points(
        list_header_center_points.begin(),
        list_header_center_points.begin() + 21
    );
    st::define_address(sliced_points, AD_frame, AD_coord);

    st::print(AD_coord);
    st::print("AD_frame: ");
    st::print(AD_frame);

    return 0;
}


int main() //test detect sequenceframes
{
    // Loading AI model
    string model_path = "model/detecting_header_model.onnx";
    st::loading_model(model_path);
    string video_path = "testing_samples/20240320_038_cut.mp4";
    std::vector<std::vector<float>> list_header_center_points;
    int start_frame = 5;
    int end_frame = 10;
    bool saveFrame = true;
    bool detect = st::detecting_header_by_sequenceframes(video_path, list_header_center_points, start_frame, end_frame, saveFrame);
    //st::loading_model(model_path);
    //string video_path2 = "testing_samples/20240320_008.mp4";
    //detect = st::detecting_header_by_sequenceframes(video_path2, list_header_center_points, start_frame, end_frame, saveFrame);
    st::print(detect);
    st::print(list_header_center_points);
    return 0;
}


