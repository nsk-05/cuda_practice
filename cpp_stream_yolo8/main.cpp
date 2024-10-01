#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/home/nsk/ultralytics"; // Set your ultralytics base path
    
    // Open the default camera (usually camera index 0)
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    cv::Mat frame;
    
    bool runOnGPU = true;

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/assets/bus.jpg");
    imageNames.push_back(projectBasePath + "/assets/zidane.jpg");

    while (true) {
        // Capture a frame from the camera
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        // Exit loop if 'q' is pressed
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();

    return 0;
}