#include "shot_trace.h"


int main()
{
    // Loading AI model
    string model_path = "model/detecting_header_model.onnx";
    st::loading_model(model_path);
    // Getting the_first_golfball_point 
    string video_path = "testing_samples/20240320_038_cut.mp4"; //20240320_038_cut GolfswingHD.mp4
    std::vector<float> the_first_golfball_point;
    st::setting_first_position_of_golfball(video_path,
        the_first_golfball_point,
        24, 0.5);
    cout << "the_first_golfball_point: " << the_first_golfball_point[0] << ", " << the_first_golfball_point[1] << endl;
    // Run Shot trace process
    string shot_trace_output_video_path = "shot_trace_video_output_0830.mp4";
    int write_fps = 10;
    int debug_header = false;
    int debug_ball = false;
    string trace_image_path = "trace_image.png";
    st::creating_shot_trace(video_path,
        shot_trace_output_video_path,
        trace_image_path,
        the_first_golfball_point,
        write_fps, debug_header, debug_ball);
    return 0;
}