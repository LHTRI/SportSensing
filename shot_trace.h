#pragma once
#ifndef SHOT_TRACE_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define SHOT_TRACE_H

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
#include "shot_trace_utils.h"
#include <numeric>


namespace st
{
    // Namespaces.
    using namespace cv;
    using namespace cv::dnn;
    using namespace std;

    // Constants.
    const float DIM = 448;
    const float INPUT_WIDTH = DIM;
    const float INPUT_HEIGHT = DIM;
    const float SCORE_THRESHOLD = 0.1;
    const float MIN_INTER_RATE = 0.2;
    const int CLASSES_COUNT = 2;

    // Text parameters.
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 1;

    // Colors.
    cv::Scalar BLACK = cv::Scalar(0, 0, 0);
    cv::Scalar BLUE = cv::Scalar(255, 0, 0);
    cv::Scalar GREEN = cv::Scalar(0, 255, 0);
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
    cv::Scalar RED = cv::Scalar(0, 0, 255);

    // AI Model
    Net HEADER_MODEL;
    
    // Other
    int FMT = VideoWriter::fourcc('m', 'p', '4', 'v');
    const float  PI_F = 3.14159265f;


    void draw_label(cv::Mat& input_image, std::string label, int left, int top)
    {
        // Display the label at the top of the bounding box.
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
        top = std::max(top, label_size.height);
        // Top left corner.
        cv::Point tlc = cv::Point(left, top - label_size.height - baseLine - 2 * THICKNESS);
        // Bottom right corner.
        cv::Point brc = cv::Point(left + label_size.width, top - 2 * THICKNESS);
        // Draw black rectangle.
        rectangle(input_image, tlc, brc, BLACK, cv::FILLED);
        // Put the label on the black rectangle.
        cv::putText(input_image, label, cv::Point(left, top - baseLine - THICKNESS), FONT_FACE, FONT_SCALE, YELLOW, 2 * THICKNESS);
    }


    float intersection_rate(cv::Rect club, cv::Rect header)
    {
        if (header.width == 0)
            return 0.0;
        else
        {
            cv::Rect intersection = club & header;
            float rate = intersection.area() / float(header.width * header.height);
            return rate;
        }
    }


    void loading_model(string model_path)
    {
        HEADER_MODEL = readNetFromONNX(model_path);
        if (HEADER_MODEL.empty())
        {
            std::cerr << "Failed to load the network. Check the model path or format." << std::endl;
        }
        else
            cout << "Successfully loaded the model!" << endl;
    }


    cv::Mat preprocessing_frame(Ptr<BackgroundSubtractor> KNN_BACKSUB,
        int LENGTH, int FRAME_WIDTH, int FRAME_HEIGHT,
        cv::Mat original_frame, bool mix_mask)
    {
        cv::Mat gray_image;
        cv::cvtColor(original_frame, gray_image, cv::COLOR_BGR2GRAY);
        std::vector<cv::Mat> channels;
        if (mix_mask)
        {
            cv::Mat mask_frame;
            KNN_BACKSUB->apply(original_frame, mask_frame);
            mask_frame.setTo(0, mask_frame == 127);
            channels = { gray_image, gray_image, mask_frame };
        }
        else
        {
            channels = { gray_image, gray_image, gray_image };
        }
        cv::Mat stack_image;
        cv::merge(channels, stack_image);
        cv::Mat image = cv::Mat::zeros(LENGTH, LENGTH, CV_8UC3);
        //cout << "check here" << endl;
        cv::Rect roi(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        stack_image.copyTo(image(roi));
        return image;
    }


    void draw_bounding_box(cv::Mat& draw_frame, cv::Rect club_box, cv::Rect header_box,
        std::vector<float> center_point, float club_score, float header_score, bool showClub)
    {
        if (showClub)
        {
            rectangle(draw_frame, club_box, BLUE, 2 * THICKNESS);
            std::string club_label = cv::format("%.2f", club_score);
            club_label = "club:" + club_label;
            draw_label(draw_frame, club_label, club_box.br().x, club_box.br().y);
        }
        rectangle(draw_frame, header_box, GREEN, 2 * THICKNESS);
        std::string header_label = cv::format("%.2f", header_score);
        header_label = "header:" + header_label;
        draw_label(draw_frame, header_label, header_box.tl().x, header_box.tl().y);
        cv::circle(draw_frame, cv::Point(center_point[0], center_point[1]), 7, Scalar(0, 255, 0), -1);
    }


    void detecting_header_byframe(cv::Mat preprocessed_frame,
        cv::Rect& best_club_box, cv::Rect& best_header_box,
        float& best_club_score, float& best_header_score,
        std::vector<float>& club_center_point,
        std::vector<float>& header_center_point, 
        float SCALE_FACTOR)
    {
        cv::Mat blob;
        blobFromImage(preprocessed_frame, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), //1. / 255. INPUT_WIDTH, INPUT_HEIGHT
            cv::Scalar(), true, false);
        
        ///Debug
        cout << "check preprocessed_frame: " << preprocessed_frame.size() << endl;
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(preprocessed_frame, &minVal, &maxVal);
        std::cout << "preprocessed_frame Min value: " << minVal << endl;
        std::cout << "preprocessed_frame Max value: " << maxVal << endl;

        std::cout << "blob.dims: " << blob.dims << std::endl;
        std::cout << "blob.size: " << blob.size << std::endl;

        HEADER_MODEL.setInput(blob);
        std::vector<cv::Mat> outputs;
        HEADER_MODEL.forward(outputs, HEADER_MODEL.getUnconnectedOutLayersNames());

        int rows = outputs[0].size[2];
        int dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]); //[84 x 8400]
        auto data = (float*)outputs[0].data;
        //std::cout << "check data info: " << rows << ", " << dimensions << endl;

        for (int i = 0; i < rows; ++i)
        {
            float* classes_scores = data + 4;
            cv::Mat scores(1, CLASSES_COUNT, CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id);
            if (maxClassScore > SCORE_THRESHOLD)
            {
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * SCALE_FACTOR);
                int top = int((y - 0.5 * h) * SCALE_FACTOR);

                int width = int(w * SCALE_FACTOR);
                int height = int(h * SCALE_FACTOR);

                if (class_id.x == 0 && maxClassScore > best_club_score)
                {
                    best_club_score = maxClassScore;
                    best_club_box = cv::Rect(left, top, width, height);
                }
                if (class_id.x == 1 && maxClassScore > best_header_score)
                {
                    best_header_score = maxClassScore;
                    best_header_box = cv::Rect(left, top, width, height);
                }
            }
            data += dimensions;
        }
        club_center_point = std::vector<float>{ float(best_club_box.x + 0.5 * best_club_box.width),
            float(best_club_box.y + 0.5 * best_club_box.height) };
        header_center_point = std::vector<float>{ float(best_header_box.x + 0.5 * best_header_box.width),
            float(best_header_box.y + 0.5 * best_header_box.height) };
    }


    void detecting_rawPhases(bool& bool_AD, bool& bool_TP, bool& bool_IM, bool& bool_FN,
        std::vector<float> club_center_point, std::vector<float> header_center_point)
    {
        if (header_center_point[0] == 0)
            return;
        if (!bool_AD && header_center_point[0] > club_center_point[0] && header_center_point[1] > club_center_point[1])
        {
            cout << "Raw Address detected" << endl;
            bool_AD = true;
        }
        if (bool_AD && !bool_TP && header_center_point[0] < club_center_point[0] && header_center_point[1] < club_center_point[1])
        {
            bool_TP = true;
            cout << "Raw Top detected" << endl;
        }
        if (bool_TP && !bool_IM && header_center_point[0] > club_center_point[0] && header_center_point[1] > club_center_point[1])
        {
            bool_IM = true;
            cout << "Raw IM detected" << endl;
        }
        if (bool_IM && !bool_FN && header_center_point[0] < club_center_point[0] && header_center_point[1] < club_center_point[1])
        {
            bool_FN = true;
            cout << "Raw FN detected" << endl;
        }
    }


    void locating_current_region(int& current_region, std::vector<float> club_center_point,
        std::vector<float> header_center_point)
    {
        if (header_center_point[0] > club_center_point[0] && header_center_point[1] > club_center_point[1])
        {
            current_region = 1;
        }
        else if (header_center_point[0] < club_center_point[0] && header_center_point[1] > club_center_point[1])
        {
            current_region = 2;
        }
        else if (header_center_point[0] < club_center_point[0] && header_center_point[1] < club_center_point[1])
        {
            current_region = 3;
        }
        else if (header_center_point[0] > club_center_point[0] && header_center_point[1] < club_center_point[1])
        {
            current_region = 4;
        }
    }


    void redetecting_header(std::vector<float>& header_center_point, cv::Rect& best_club_box, cv::Rect& best_header_box,
        int current_region, std::vector<int> header_size)
    {
        if (best_club_box.width != 0)
        {
            int left = 0, top = 0;
            if (current_region == 1)
            {
                left = best_club_box.br().x - header_size[0];
                top = best_club_box.br().y - header_size[1];
            }
            else if (current_region == 2)
            {
                left = best_club_box.tl().x;
                top = best_club_box.br().y - header_size[1];
            }
            else if (current_region == 3)
            {
                left = best_club_box.tl().x;
                top = best_club_box.tl().y;
            }
            else if (current_region == 4)
            {
                left = best_club_box.br().x - header_size[0];
                top = best_club_box.tl().y;
            }
            best_header_box = cv::Rect(left, top, header_size[0], header_size[1]);
            header_center_point = std::vector<float>{ float(best_header_box.x + 0.5 * best_header_box.width),
                float(best_header_box.y + 0.5 * best_header_box.height) };
        }
    }


    void detecting_header_byvideo(string video_path, string saved_output_video_path,
        int write_fps, bool showClub, bool show_stack_input, bool saveFrame,
        std::vector<std::vector<float>>& list_header_center_points,
        std::vector<float>& AD_wrist_position)
    {
        string saved_output_frame = "./saved_header_detection_frames";
        if (saveFrame)
        {
            utils::fs::remove_all(saved_output_frame);
            utils::fs::createDirectory(saved_output_frame);
        }

        VideoCapture try_cap(video_path);
        Mat try_frame;
        try_cap.read(try_frame);
        try_cap.release();
        int FRAME_WIDTH = try_frame.cols;
        int FRAME_HEIGHT = try_frame.rows;
        cout << "check FRAME_WIDTH: " << FRAME_WIDTH << endl;
        cout << "check FRAME_HEIGHT: " << FRAME_HEIGHT << endl;
        VideoCapture cap(video_path);

        int LENGTH = std::max(FRAME_WIDTH, FRAME_HEIGHT);
        float SCALE_FACTOR = LENGTH / DIM;
        VideoWriter writer;
        if (show_stack_input)
            writer = VideoWriter(saved_output_video_path, FMT, write_fps, Size(LENGTH, LENGTH));
        else
            writer = VideoWriter(saved_output_video_path, FMT, write_fps, Size(FRAME_WIDTH, FRAME_HEIGHT));
        Mat frame;
        int frame_num = 0;
        bool bool_AD = false;
        bool bool_TP = false;
        bool bool_IM = false;
        bool bool_FN = false;
        std::vector<int> header_size(0, 0);
        int current_region = 0;
        bool detectFailed = false;
        std::vector<float> previous_header_center_point;
        AD_wrist_position = std::vector<float>{ 0., 0. };

        Ptr<BackgroundSubtractor> KNN_BACKSUB = createBackgroundSubtractorKNN(3, 500, true);

        while (cap.read(frame))
        {
            cv::Mat preprocessed_frame;
            preprocessed_frame = preprocessing_frame(KNN_BACKSUB, LENGTH, FRAME_WIDTH, FRAME_HEIGHT, frame, true);
            //cout << "check here" << endl;
            cv::Rect best_club_box(0, 0, 0, 0), best_header_box(0, 0, 0, 0);
            float best_club_score = 0.;
            float best_header_score = 0.;
            std::vector<float> club_center_point, header_center_point;
            detecting_header_byframe(preprocessed_frame,
                best_club_box, best_header_box,
                best_club_score, best_header_score,
                club_center_point, header_center_point, SCALE_FACTOR);
            std::cout << frame_num << " :header_center_point: " << header_center_point[0] << endl;
            float rate = intersection_rate(best_club_box, best_header_box);
            if (rate < MIN_INTER_RATE)
                std::cout << frame_num << " :No detection" << endl;
            else
            {
                header_size = std::vector<int>{ best_header_box.width, best_header_box.height };
                locating_current_region(current_region, club_center_point, header_center_point);
                std::cout << "current_region: " << current_region << endl;
            }
            if (rate == 0 || best_header_score == 0)
            {
                detectFailed = true;
                std::cout << frame_num << " :redetecting header: " << endl;
                redetecting_header(header_center_point, best_club_box, best_header_box, current_region, header_size);
            }

            detecting_rawPhases(bool_AD, bool_TP, bool_IM, bool_FN,
                club_center_point, header_center_point);
            if (bool_FN)
                break;
            if (bool_AD && AD_wrist_position[0] == 0)
                AD_wrist_position = std::vector<float>{ (float)best_club_box.x, (float)best_club_box.y };
            
            if (header_center_point[0] != 0)
            {
                previous_header_center_point = header_center_point;
            }
            else
                header_center_point = previous_header_center_point;
            list_header_center_points.push_back(header_center_point);
            cv::Mat draw_frame;
            if (show_stack_input)
                draw_frame = preprocessed_frame.clone();
            else
                draw_frame = frame.clone();
            draw_bounding_box(draw_frame, best_club_box, best_header_box,
                header_center_point, best_club_score, best_header_score, showClub);
            cv::imshow("Detecting header", draw_frame);
            if (saveFrame)
            {
                string file = to_string(frame_num) + ".png";
                string save_path = utils::fs::join(saved_output_frame, file);
                imwrite(save_path, draw_frame);
            }
            writer.write(draw_frame);
            int k = waitKey(1);
            if (k == 27)
            {
                break;
            }
            frame_num += 1;
        }
        cap.release();
        writer.release();
        cv::destroyAllWindows();
    }


    void define_phases(std::vector<std::vector<float>> list_header_center_points,
        std::vector<std::vector<float>>& trimmed_header_center_points,
        int& AD_frame, int& TOP_frame, int& IM_frame)
    {
        int raw_top = std::distance(list_header_center_points.begin(),
            std::min_element(list_header_center_points.begin(),
                list_header_center_points.end() - 10,
                [](const std::vector<float>& a, const std::vector<float>& b)
                {
                    return a[1] < b[1];
                }
            )
        );
        std::vector<float> IM_stage = st::np_slice(list_header_center_points, raw_top, list_header_center_points.size() - 1, 1);
        //std::vector<std::vector<double>> list{ {1, 20},  {2, 30}, {3, 40}, {4, 50} };
        //std::vector<double> test = np_slice(list, 1, 2, 1);
        //cout << "check gggggggggggggggggfffffffffffffffffffffffffffffffffffffffffffffffff: " << test[0] << endl;
        IM_frame = st::np_argmax(IM_stage);
        IM_frame += raw_top + 1;
        std::vector<float> new_stage = st::np_slice(list_header_center_points, raw_top, IM_frame + 1, 1);
        new_stage = st::np_slice(new_stage, 0, int(2 * new_stage.size() / 3));
        TOP_frame = st::np_argmax(new_stage);
        TOP_frame += raw_top;
        std::vector<float> AD_stage = st::np_slice(list_header_center_points, 0, raw_top, 1);
        std::reverse(AD_stage.begin(), AD_stage.end());
        std::vector<float> delta(AD_stage.size() - 1);
        for (size_t i = 1; i < AD_stage.size(); ++i)
        {
            delta[i - 1] = AD_stage[i] - AD_stage[i - 1];
        }
        std::vector<int> AD = st::np_where(delta, st::np_greater, 10.);
        std::vector<float> delta_indices(AD.size() - 1);
        for (size_t i = 1; i < AD.size(); ++i)
        {
            delta_indices[i - 1] = AD[i] - AD[i - 1];
        }
        std::vector<int> indices = st::np_where(delta_indices, st::np_greater, 5.);
        if (indices.size() > 0)
            AD_frame = AD_stage.size() - indices[0] - 2;
        else
            AD_frame = AD.back();
        //trimmed_header_center_points = st::np_slice(list_header_center_points, AD_frame, IM_frame + 1);
        //throw std::out_of_range("Out of range error!");
    }


    void setting_first_position_of_golfball(string video_path,
        std::vector<float>& the_first_golfball_point,
        int delay, float resize_factor)
    {
        cv::VideoCapture cap(video_path, cv::CAP_FFMPEG);
        //check backend
        int backend = static_cast<int>(cap.get(cv::CAP_PROP_BACKEND));
        std::cout << "Backend ID: " << backend << std::endl;
        if (!cap.isOpened())
        {
            std::cerr << "Error opening video file" << std::endl;
            return;
        }
        cv::Mat frame, resize_frame;
        cv::Rect init_bbox;
        string window_name = "Setting the first position of golfball";
        while (true)
        {
            bool ret = cap.read(frame);
            if (!ret || frame.empty()) {
                std::cout << ret << std::endl;
                break;
            }

            cv::resize(frame, resize_frame, cv::Size(), resize_factor, resize_factor, cv::INTER_AREA);
            cv::imshow(window_name, resize_frame);
            int k = cv::waitKey(delay);

            if (k == 32) // Space key
            {
                init_bbox = cv::selectROI(window_name, resize_frame);
                if (init_bbox.width != 0 && init_bbox.height != 0)
                {
                    init_bbox.x = static_cast<int>(init_bbox.x / resize_factor);
                    init_bbox.y = static_cast<int>(init_bbox.y / resize_factor);
                    init_bbox.width = static_cast<int>(init_bbox.width / resize_factor);
                    init_bbox.height = static_cast<int>(init_bbox.height / resize_factor);
                    break;
                }
            }
            if (k == 27)
            {
                break;
            }
        }
        cap.release();
        the_first_golfball_point = std::vector<float>{ (float)(init_bbox.x + 0.5 * init_bbox.width),
                                                        (float)(init_bbox.y + 0.5 * init_bbox.height) };
        st::write_data2("the_first_golfball_point.txt", the_first_golfball_point);
        //cv::rectangle(frame, init_bbox, BLUE, 2 * THICKNESS);
        //cv::resize(frame, frame, cv::Size(700, 700));
        //cv::imshow(window_name, frame);
        //cv::waitKey(0);
        cv::destroyAllWindows();
        cout << "init box: " << init_bbox << endl;
    }


    void testing_detect_header(string video_path)
    {
        string model_path = "model/detecting_header_model.onnx";
        loading_model(model_path);
        string saved_output_video_path = "testing_header_detection_video.mp4";
        int write_fps = 10;
        bool show_stack_input = true;
        bool saveFrame = true;
        std::vector<std::vector<float>> list_header_center_points;
        std::vector<float> AD_wrist_position;
        detecting_header_byvideo(video_path, saved_output_video_path, write_fps, true, show_stack_input, saveFrame,
            list_header_center_points, AD_wrist_position);
        cout << "AD_wrist_position: " << AD_wrist_position[0] << " " << AD_wrist_position[1] << endl;
        st::write_data("list_header_center_points.txt", list_header_center_points);
        st::write_data2("AD_wrist_position.txt", AD_wrist_position);
    }


    int testing_detection_with_single_frame(string frame_path)
    {
        string model_path = "model/detecting_header_model.onnx";
        loading_model(model_path);
        cv::Mat original_frame = cv::imread(frame_path);
        cv::Mat preprocessed_frame;
        int FRAME_WIDTH = original_frame.cols;
        int FRAME_HEIGHT = original_frame.rows;
        int LENGTH = std::max(FRAME_WIDTH, FRAME_HEIGHT);
        float SCALE_FACTOR = LENGTH / DIM;
        cout << LENGTH << ", " << SCALE_FACTOR << endl;
        Ptr<BackgroundSubtractor> KNN_BACKSUB = createBackgroundSubtractorKNN(3, 500, true);
        preprocessed_frame = preprocessing_frame(KNN_BACKSUB, LENGTH, FRAME_WIDTH, FRAME_HEIGHT, original_frame, true);
        cout << preprocessed_frame.size() << endl;
        cv::Rect best_club_box(0, 0, 0, 0), best_header_box(0, 0, 0, 0);
        float best_club_score = 0.;
        float best_header_score = 0.;
        std::vector<float> club_center_point;
        std::vector<float> header_center_point;
        detecting_header_byframe(preprocessed_frame,
            best_club_box, best_header_box,
            best_club_score, best_header_score,
            club_center_point, header_center_point, SCALE_FACTOR);
        cv::Mat draw_frame = original_frame.clone();
        draw_bounding_box(draw_frame, best_club_box, best_header_box,
            header_center_point, best_club_score, best_header_score, true);
        cv::imwrite("draw_frame.png", draw_frame);
        cv::resize(draw_frame, draw_frame, cv::Size(700, 700));
        cv::imshow("Output", draw_frame);
        cv::waitKey(0);
        return 0;
    }


    int testing_detect_phases()
    {
        std::vector<std::vector<float>> list_header_center_points, trimmed_header_center_points;
        std::string filename = "list_header_center_points.txt";
        st::read_data(filename, list_header_center_points);
        int AD_frame, TOP_frame, IM_frame;
        define_phases(list_header_center_points, trimmed_header_center_points,
            AD_frame, TOP_frame, IM_frame);
        cout << "AD_frame: " << AD_frame << endl;
        cout << "TOP_frame: " << TOP_frame << endl;
        cout << "IM_frame: " << IM_frame << endl;
        st::write_data("trimmed_header_center_points.txt", trimmed_header_center_points);
    }


    Ptr<SimpleBlobDetector> create_ball_detector(double minConvexity,
        double minCircularity,
        double minInertiaRatio)
    {
        double minRadius = 1;
        double maxRadius = 30 * minRadius;

        cv::SimpleBlobDetector::Params params;
        params.minThreshold = 150;
        params.maxThreshold = 255;
        params.filterByArea = true;
        params.minArea = 1;
        params.maxArea = PI_F * maxRadius * maxRadius;
        params.blobColor = 255;

        // Filter by Convexity
        params.filterByConvexity = true;
        params.minConvexity = minConvexity;
        params.maxConvexity = 1.1;

        // Filter by Circularity
        params.filterByCircularity = true;
        params.minCircularity = minCircularity;
        params.maxCircularity = 1.1;

        // Filter by Inertia
        params.filterByInertia = true;
        params.minInertiaRatio = minInertiaRatio;
        params.maxInertiaRatio = 1.1;

        return cv::SimpleBlobDetector::create(params);
    }


    void detecting_the_best_reference_point(string video_path,
        std::vector<float>& the_best_reference_point,
        std::vector<float> the_first_golfball_point,
        std::vector<float> AD_wrist_position,
        int IM_frame, bool save_output)
    {
        Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorKNN(3, 500, true);
        //Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2(20, 100, true);
        Ptr<SimpleBlobDetector> golfball_detector = create_ball_detector(0.2, 0.2, 0.1);
        string check_output_folder = "./check_outputs_1";
        string check_mask_folder = "./check_masks_1";
        if (save_output)
        {
            utils::fs::remove_all(check_output_folder);
            utils::fs::createDirectory(check_output_folder);
            utils::fs::remove_all(check_mask_folder);
            utils::fs::createDirectory(check_mask_folder);
        }
        std::vector<float> previous_center_point = the_first_golfball_point;
        int left_angle = 35;
        int right_angle = -45;
        int number_frame = 3;
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
            if (frame_num >= IM_frame && frame_num <= IM_frame + number_frame)
            {
                std::vector<KeyPoint> keypoints;
                golfball_detector->detect(mask_frame, keypoints);
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
                        if (center_point.y < AD_wrist_position[1])
                        {
                            cout << "Removed by AD_wrist_position" << endl;
                            continue;
                        }
                        std::vector<float> vect = { the_first_golfball_point[0] - center_point.x,
                                                      the_first_golfball_point[1] - center_point.y };
                        float angle = std::atan2(vect[0], vect[1]);
                        angle = angle * (180.0 / PI_F);
                        //cout << "angle: " << angle << endl;
                        if (angle > left_angle || angle < right_angle)
                        {
                            cout << "Removed by boundary angle" << endl;
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
                previous_center_point = selected_center_point;
                the_best_reference_point = selected_center_point;
                cv::circle(frame, Point(selected_center_point[0], selected_center_point[1]), int(selected_rad), Scalar(255, 0, 0), 2);
                //cv::imshow("frame", frame);
                //cv::imshow("mask", mask_frame);
                string file = to_string(frame_num) + ".png";
                string save_path = utils::fs::join(check_mask_folder, file);
                imwrite(save_path, mask_frame);
                save_path = utils::fs::join(check_output_folder, file);
                imwrite(save_path, frame);
            }
            int k = waitKey(1);
            if (k == 27 || frame_num > IM_frame + number_frame)
            {
                break;
            }
            frame_num++;
        }
        cap.release();
        cv::destroyAllWindows();
    }


    void detecting_golfball(string video_path,
        std::vector<std::vector<float>>& list_golfball_center_points,
        std::vector<float> the_first_golfball_point,
        std::vector<float> the_best_reference_point,
        int IM_frame, bool save_output)
    {
        std::vector<float> vect = { the_first_golfball_point[0] - the_best_reference_point[0],
                                    the_first_golfball_point[1] - the_best_reference_point[1] };
        float std_angle = std::atan2(vect[0], vect[1]);
        std_angle = std_angle * (180.0 / PI_F);
        cout << "check std_angle: " << std_angle << endl;
        float max_dis = std::sqrt(vect[0] * vect[0] + vect[1] * vect[1]);
        Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorKNN(3, 500, true);
        //Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2(20, 100, true);
        //Ptr<BackgroundSubtractor> backSub = createBackgroundSubtractorMOG2();
        Ptr<SimpleBlobDetector> golfball_detector = create_ball_detector(0.1, 0.1, 0.01);
        string check_output_folder = "./check_outputs_2";
        string check_mask_folder = "./check_masks_2";
        if (save_output)
        {
            utils::fs::remove_all(check_output_folder);
            utils::fs::createDirectory(check_output_folder);
            utils::fs::remove_all(check_mask_folder);
            utils::fs::createDirectory(check_mask_folder);
        }
        std::vector<float> previous_center_point = the_first_golfball_point;
        int count_ball = 0;
        int count_undetect = 0;
        int count_undetect_max = 3;
        float max_tolerance_angle = 2;
        float mean_ball_radius = 0;
        std::vector<float> list_ball_radius;
        std::vector<float> list_sequence_dis;

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
                std::vector<KeyPoint> keypoints;
                golfball_detector->detect(mask_frame, keypoints);
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
                        angle = angle * (180.0 / PI_F);
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
                list_golfball_center_points.push_back(selected_center_point);
                //cv::imshow("frame", frame);
                if (save_output)
                {
                    string file = to_string(frame_num) + ".png";
                    string save_path = utils::fs::join(check_mask_folder, file);
                    imwrite(save_path, mask_frame);
                    save_path = utils::fs::join(check_output_folder, file);
                    imwrite(save_path, frame);
                }

            }
            int k = waitKey(1);
            if (k == 27)
            {
                break;
            }
            if (count_undetect == 3)
            {
                cout << "No need detect any more" << endl;
                break;
            }
            frame_num++;
        }
        list_golfball_center_points.insert(list_golfball_center_points.begin(), the_first_golfball_point);
        cap.release();
        cv::destroyAllWindows();
    }


    void draw_shot_trace(string video_path, string video_output_path,
        string trace_image_path,
        std::vector<std::vector<float>> list_header_center_points,
        std::vector<std::vector<float>> list_golfball_center_points,
        int IM_frame, int write_fps, int interval)
    {
        VideoCapture cap(video_path);
        int FRAME_WIDTH = cap.get(CAP_PROP_FRAME_WIDTH);
        int FRAME_HEIGHT = cap.get(CAP_PROP_FRAME_HEIGHT);
        VideoWriter writer(video_output_path, FMT, write_fps, Size(FRAME_WIDTH, FRAME_HEIGHT));
        std::vector<std::vector<float>> cut_header_center_points(list_header_center_points.begin(),
            list_header_center_points.begin() + IM_frame);
        int length_1 = cut_header_center_points.size();
        int length_2 = list_golfball_center_points.size();
        int length = length_1 + length_2;
        int number_lines_1 = 0;
        int number_lines_2 = 0;
        int frame_num = 0;
        Mat frame, trace_image;
        while (cap.read(frame))
        {
            if (frame_num >= 0 && frame_num <= length_1 - 1)
                number_lines_1 = frame_num;
            if (number_lines_1 > 0)
            {
                for (int index = 0; index < number_lines_1; ++index)
                {
                    Point2i point_1(cut_header_center_points[index][0], cut_header_center_points[index][1]);
                    Point2i point_2(cut_header_center_points[index + 1][0], cut_header_center_points[index + 1][1]);
                    cv::line(frame, point_1, point_2, cv::Scalar(0, 255, 0), 7, cv::LINE_AA);
                }
            }
            if (frame_num >= IM_frame && frame_num <= IM_frame + length_2 - 1)
                number_lines_2 = frame_num - IM_frame;
            if (number_lines_2 > 0)
            {
                for (int index = 0; index < number_lines_2; ++index)
                {
                    Point2i point_1(list_golfball_center_points[index][0], list_golfball_center_points[index][1]);
                    Point2i point_2(list_golfball_center_points[index + 1][0], list_golfball_center_points[index + 1][1]);
                    cv::line(frame, point_1, point_2, cv::Scalar(0, 0, 255), 7, cv::LINE_AA);
                }
            }
            cv::imshow("Shot trace", frame);
            writer.write(frame);
            trace_image = frame;
            int k = waitKey(interval);
            if (k == 27)
            {
                break;
            }
            frame_num++;
        }
        cv::imwrite(trace_image_path, trace_image);
        cap.release();
        writer.release();
        cv::destroyAllWindows();
    }


    int testing_detect_golfball(string video_path, int IM_frame)
    {
        std::vector<float> the_first_golfball_point;
        st::read_data2("the_first_golfball_point.txt", the_first_golfball_point);
        cout << "the_first_golfball_point: " << the_first_golfball_point[0] << ", " << the_first_golfball_point[1] << endl;
        std::vector<float> AD_wrist_position;
        st::read_data2("AD_wrist_position.txt", AD_wrist_position);
        cout << "AD_wrist_position: " << AD_wrist_position[0] << ", " << AD_wrist_position[1] << endl;
        std::vector<float> the_best_reference_point;
        detecting_the_best_reference_point(video_path,
            the_best_reference_point,
            the_first_golfball_point,
            AD_wrist_position,
            IM_frame, true);
        cout << "the_best_reference_point: " << the_best_reference_point[0] << ", " << the_best_reference_point[1] << endl;
        std::vector<std::vector<float>> list_golfball_center_points;
        detecting_golfball(video_path,
            list_golfball_center_points,
            the_first_golfball_point,
            the_best_reference_point,
            IM_frame, true);
        st::write_data("list_golfball_center_points.txt", list_golfball_center_points);
        return 0;
    }


    void testing_draw_shot_trace()
    {
        string video_path = "testing_samples/20240320_038_cut.mp4";
        string video_output_path = "shot_trace.mp4";
        int IM_frame = 63;
        int write_fps = 10;
        std::vector<std::vector<float>> list_header_center_points;
        st::read_data("list_header_center_points.txt", list_header_center_points);
        std::vector<std::vector<float>> list_golfball_center_points;
        st::read_data("list_golfball_center_points.txt", list_golfball_center_points);
        string trace_image_path = "trace_image.png";
        draw_shot_trace(video_path, video_output_path,
            trace_image_path,
            list_header_center_points,
            list_golfball_center_points,
            IM_frame, write_fps, 24);
    }


    void creating_shot_trace(string video_path,
        string shot_trace_output_video_path,
        string trace_image_path,
        std::vector<float> the_first_golfball_point,
        int write_fps, bool debug_header, bool debug_ball)
    {
        // Detect header
        bool show_stack_input = true;
        bool showClub = true;
        std::vector<std::vector<float>> list_header_center_points;
        std::vector<float> AD_wrist_position;
        string saved_output_video_path = "header_detection_video_output.mp4";
        detecting_header_byvideo(video_path, saved_output_video_path, write_fps,
            showClub, show_stack_input, debug_header,
            list_header_center_points, AD_wrist_position);
        //Check leng list_header_center_points
        cout << "Check leng list_header_center_points: " << list_header_center_points.size() << endl;
        if (list_header_center_points.size() < 15)
        {
            cout << "Detect header failed" << endl;
            return;
        }
        // Finding phases
        std::vector<std::vector<float>> trimmed_header_center_points;
        int AD_frame, TOP_frame, IM_frame;
        try 
        {
            define_phases(list_header_center_points, trimmed_header_center_points,
                AD_frame, TOP_frame, IM_frame);
            //throw std::out_of_range("Out of range error!");
        }
        catch (...)
        {
            std::cerr << "Caught an unknown exception in define_phases!" << std::endl;
            return;
        }
        cout << "AD_frame, TOP_frame, IM_frame: " << AD_frame << ", " << TOP_frame << ", " << IM_frame << endl;
        // Detect golfball
        std::vector<float> the_best_reference_point;
        cout << "************ Starting detect golfball *************" << endl;
        detecting_the_best_reference_point(video_path,
            the_best_reference_point,
            the_first_golfball_point,
            AD_wrist_position,
            IM_frame, debug_ball);
        std::vector<std::vector<float>> list_golfball_center_points;
        cout << "the_best_reference_point: " << the_best_reference_point[0] << ", " << the_best_reference_point[1] << endl;
        detecting_golfball(video_path,
            list_golfball_center_points,
            the_first_golfball_point,
            the_best_reference_point,
            IM_frame, debug_ball);
        cout << "list_golfball_center_points size: " << list_golfball_center_points.size() << endl;
        // Draw result
        draw_shot_trace(video_path,
            shot_trace_output_video_path,
            trace_image_path,
            list_header_center_points,
            list_golfball_center_points,
            IM_frame, write_fps, 24);
    }

    //Update 2024/11/07
    void draw_ball_trajectory(string video_path,
        string trajectory_data_path,
        int IM_frame, int write_fps, int interval)
    {
        std::vector<std::vector<float>> list_ball_center_points;
        st::read_data(trajectory_data_path, list_ball_center_points);
        
        VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "Error opening video stream or file" << endl;
            return;
        }
        int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
        int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));

        int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
        string save_file = std::filesystem::path(video_path).stem().string() + "_ball_trace.mp4";
        string save_folder = "ball_trace_outputs";

        if (!std::filesystem::exists(save_folder))
        {
            std::filesystem::create_directories(save_folder);
        }
        std::filesystem::path save_path = std::filesystem::path(save_folder) / save_file;
        //cout << save_path.string();
        VideoWriter writer(save_path.string(), fourcc, write_fps, Size(frame_width, frame_height), true);
        
        int length_2 = list_ball_center_points.size();
        int number_lines_2 = 0;

        int frame_num = 0;
        Mat frame, previous_frame;
        bool finish = false;

        while (true) {
            bool ret = cap.read(frame);
            if (!ret || frame.empty()) {
                cout << "Real video finished" << endl;
                finish = true;
            }
            if (!finish) {
                previous_frame = frame.clone();
            }
            else {
                cout << "Extra video" << endl;
                frame = previous_frame.clone();
            }

            // Visual ball trace
            if (frame_num >= IM_frame && frame_num <= IM_frame + length_2 - 1) {
                number_lines_2 = frame_num - IM_frame;
            }
            if (number_lines_2 > 0) {
                for (int index = 0; index < number_lines_2; ++index) {
                    Point point_1(list_ball_center_points[index][0], list_ball_center_points[index][1]);
                    Point point_2(list_ball_center_points[index+1][0], list_ball_center_points[index+1][1]);
                    line(frame, point_1, point_2, Scalar(0, 255, 0), 7, LINE_AA);
                }
            }

            writer.write(frame);
            imshow("frame", frame);
            char k = waitKey(interval);
            if (k == 27) {
                break;
            }
            if (frame_num > IM_frame + length_2 && finish) {
                break;
            }
            frame_num++;
        }

        save_file = std::filesystem::path(video_path).stem().string() + "_ball_trace.png";
        save_path = std::filesystem::path(save_folder) / save_file;
        imwrite(save_path.string(), frame);

        destroyAllWindows();
        cap.release();
        writer.release();
    }


    std::vector<float> data_filter(const std::vector<float> data, int kernel_size, string mode = "median")
    {
        std::vector<float> filtered_data(data.size(), 0.0);
        int padding = kernel_size / 2;

        for (size_t i = padding; i < data.size() - padding; ++i) {
            std::vector<float> window(data.begin() + i - padding, data.begin() + i + padding + 1);
            if (mode == "median")
            {
                std::sort(window.begin(), window.end());
                filtered_data[i] = window[window.size() / 2];
            }
            else
                filtered_data[i] = std::accumulate(window.begin(), window.end(), 0.0) / window.size();
        }
        std::vector<float> res(filtered_data.begin() + padding, filtered_data.end() - padding);
        return res;
    }


    template <typename T>
    void print(T data)
    {
        std::cout << data << endl;
    }


    template <typename T>
    void print(std::vector<T> data)
    {
        for (T value : data)
        {
            std::cout << value << " ";
        }
        std::cout << endl;
    }


    template <typename T>
    void print(std::vector<std::vector<T>> data)
    {
        for (std::vector<T> value : data)
        {
            print(value);
        }
    }


    bool define_address(const std::vector<std::vector<float>> list_header_center_points,
        int& AD_frame, std::vector<int>& AD_coord)
    {
        int length = list_header_center_points.size();
        if (length < 3)
        {
            cout << "Not enough data to define the Address" << endl;
            return false;
        }
        // Split X và Y
        std::vector<float> x_coords, y_coords;
        for (const auto& point : list_header_center_points)
        {
            x_coords.push_back(point[0]);
            y_coords.push_back(point[1]);
        }
        //cout << "y_coords" << endl;
        //print(y_coords);
        // np.argmax
        auto max_index_iter = std::max_element(y_coords.begin(), y_coords.end());
        size_t max_index = std::distance(y_coords.begin(), max_index_iter);
        std::vector<float> survey_section_X(x_coords.begin(), x_coords.begin() + max_index);
        std::vector<float> survey_section_Y(y_coords.begin(), y_coords.begin() + max_index);
        float AD_Xcoord, AD_Ycoord;
        if (length > 10)
        {
            std::vector<float> filtered_Xdata;
            std::vector<float> filtered_Ydata;
            if (length < 17)
            {
                int kernel_size = 3;
                filtered_Xdata = data_filter(survey_section_X, kernel_size, "median");
                filtered_Ydata = data_filter(survey_section_Y, kernel_size, "median");
                AD_frame = length / 2;
            }
            else
            {
                int kernel_size = 5;
                filtered_Ydata = data_filter(survey_section_Y, kernel_size, "median");
                filtered_Ydata = data_filter(filtered_Ydata, kernel_size, "avg");
                //cout << "filtered_Ydata" << endl;
                //print(filtered_Ydata);
                std::vector<float> inversed_data(filtered_Ydata.rbegin(), filtered_Ydata.rend());
                std::vector<float> delta(inversed_data.size() - 1);
                for (size_t i = 0; i < delta.size(); ++i)
                {
                    delta[i] = inversed_data[i] - inversed_data[i + 1];
                    //cout << "delta: " << inversed_data[i] - inversed_data[i + 1] << endl;
                }

                // Find the last index for which delta > 1
                auto it = std::find_if(delta.rbegin(), delta.rend(), [](float d) { return d > 1; });
                int selected_index = std::distance(delta.begin(), it.base()) - 1;
                AD_frame = filtered_Ydata.size() - selected_index;
                filtered_Xdata = data_filter(survey_section_X, kernel_size, "median");
                filtered_Xdata = data_filter(filtered_Xdata, kernel_size, "avg");
            }
            AD_Xcoord = filtered_Xdata[AD_frame];
            AD_Ycoord = filtered_Ydata[AD_frame];
        }
        else
        {
            AD_frame = length / 2;
            AD_Xcoord = st::np_median(survey_section_X);
            AD_Ycoord = st::np_median(survey_section_Y);
        }
        AD_coord = { static_cast<int>(AD_Xcoord), static_cast<int>(AD_Ycoord) };
        return true;
    }


    bool check_video(string video_path, int& FRAME_WIDTH, int& FRAME_HEIGHT)
    {
        cv::VideoCapture trycap(video_path);
        if (!trycap.isOpened())
        {
            std::cout << "Failed to read video" << std::endl;
            return false;
        }

        cv::Mat try_frame;
        bool ret = trycap.read(try_frame);

        if (!ret)
        {
            std::cout << "Failed to read video" << std::endl;
            return false;
        }
        else
        {
            trycap.release();
            FRAME_WIDTH = try_frame.cols;
            FRAME_HEIGHT = try_frame.rows;
            return true;
        }
    }


    bool detecting_header_by_sequenceframes(string video_path,
        std::vector<std::vector<float>>& list_header_center_points,
        int start_frame, int end_frame, bool saveFrame=false)
    {
        string saved_output_frame = "./saved_header_detection_frames";
        if (saveFrame)
        {
            utils::fs::remove_all(saved_output_frame);
            utils::fs::createDirectory(saved_output_frame);
        }
        int back_frame = 5;
        if (start_frame > end_frame)
        {
            std::cout << "end_frame must be greater than or equal to start_frame!" << std::endl;
            return false;
        }
        if (start_frame < back_frame)
        {
            std::cout << "The smallest start_frame must be " << back_frame << std::endl;
            return false;
        }
        int FRAME_WIDTH, FRAME_HEIGHT;
        if (!check_video(video_path, FRAME_WIDTH, FRAME_HEIGHT))
            return false;
        
        cv::VideoCapture cap(video_path);

        int _startFrame = start_frame - back_frame;
        cap.set(cv::CAP_PROP_POS_FRAMES, _startFrame);

        int LENGTH = std::max(FRAME_WIDTH, FRAME_HEIGHT);
        float SCALE_FACTOR = static_cast<float>(LENGTH) / DIM;
        Ptr<BackgroundSubtractor> KNN_BACKSUB = createBackgroundSubtractorKNN(3, 500, true);
        for (int frame_num = _startFrame; frame_num < start_frame; ++frame_num)
        {
            cv::Mat frame;
            if (!cap.read(frame))
            {
                std::cout << "Failed to read frame at " << frame_num << std::endl;
                return false;
            }

            cv::Mat mask_frame;
            KNN_BACKSUB->apply(frame, mask_frame);
        }

        for (int frame_num = start_frame; frame_num < end_frame+1; ++frame_num)
        {
            cv::Mat frame;
            if (!cap.read(frame))
            {
                std::cout << "Failed to read frame at " << frame_num << std::endl;
                return false;
            }

            cv::Mat preprocessed_frame;
            preprocessed_frame = preprocessing_frame(KNN_BACKSUB, LENGTH, FRAME_WIDTH, FRAME_HEIGHT, frame, true);
            cv::Rect best_club_box(0, 0, 0, 0), best_header_box(0, 0, 0, 0);
            float best_club_score = 0.;
            float best_header_score = 0.;
            std::vector<float> club_center_point, header_center_point;
            detecting_header_byframe(preprocessed_frame,
                best_club_box, best_header_box,
                best_club_score, best_header_score,
                club_center_point, header_center_point, SCALE_FACTOR);
            list_header_center_points.push_back(header_center_point);
            if (saveFrame)
            {
                string file = to_string(frame_num) + ".png";
                string save_path = utils::fs::join(saved_output_frame, file);
                cv::circle(frame, cv::Point(header_center_point[0], header_center_point[1]), 7, Scalar(0, 255, 0), -1);
                imwrite(save_path, frame);
            }
        }
        return true;
    }
}

#endif