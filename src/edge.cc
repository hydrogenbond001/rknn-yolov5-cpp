#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <math.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <iostream>
#include <vector>

#include <errno.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <pthread.h>
#include <stdlib.h>

int fd;

void uart_init() {
    if ((fd = serialOpen("/dev/ttyS0", 9600)) < 0) {
        fprintf(stderr, "Unable to open serial device: %s\n", strerror(errno));
        return;
    }
    if (wiringPiSetup() == -1) {
        fprintf(stderr, "Unable to start wiringPi: %s\n", strerror(errno));
        return;
    }
    printf("UART initialized successfully!\n");
}

// 发送整数的函数
void serialPutNumber(int num) {
    char buffer[32];  // 定义足够大的缓冲区来存储数字字符串
    snprintf(buffer, sizeof(buffer), "%d", num);  // 将整数转换为字符串
    serialPuts(fd, buffer);  // 通过 serialPuts 发送字符串
}

// 发送浮点数的函数
void serialPutFloat(float num) {
    char buffer[32];  // 定义缓冲区来存储浮点数字符串
    snprintf(buffer, sizeof(buffer), "%.2f", num);  // 将浮点数转换为字符串，保留两位小数
    serialPuts(fd, buffer);  // 通过 serialPuts 发送字符串
}

void send_slope_data(std::vector<cv::Vec4i> lines, cv::Mat &roi) {
   // for (size_t i = 0; i <0; i++) {
	if (!lines.empty()){
        int x1 = lines[0][0];
        int y1 = lines[0][1];
        int x2 = lines[0][2];
        int y2 = lines[0][3];

        // 计算斜率
        if (x2 - x1 != 0) {
            // 计算斜率（乘以10调整到较大的范围）
            float slope = 10 * (180.0 / CV_PI) * (10.0 * (y2 - y1) / (x2 - x1));

            // 计算中点
            int midpoint_x = (x1 + x2) / 2;
            int midpoint_y = (y1 + y2) / 2;

            // 打印中点和斜率信息
            printf("%d %d %.0f\n", midpoint_x, midpoint_y, slope);

            // 发送起始字节
            unsigned char start_byte = 0x03;
            write(fd, &start_byte, 1);

            // 发送斜率数据
            char slope_buffer[32];
            snprintf(slope_buffer, sizeof(slope_buffer), "%.0f\n", slope);
            write(fd, slope_buffer, strlen(slope_buffer));

            // 发送结束字节
            unsigned char end_byte = 0xFE;
            write(fd, &end_byte, 1);
        }


        // 打印每条直线的坐标到终端
        std::cout << "Line " << 0 << ": (" 
                  << lines[0][0] << ", " << lines[0][1] << ") -> (" 
                  << lines[0][2] << ", " << lines[0][3] << ")" << std::endl;

   }
}

cv::VideoCapture init_camera(int camera_id) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        fprintf(stderr, "Error: Could not initialize camera\n");
        exit(1);
    }
    return cap;
}

void process_frame(cv::Mat frame, cv::Scalar left_lower_hsv, cv::Scalar left_upper_hsv,
                   cv::Scalar right_lower_hsv, cv::Scalar right_upper_hsv,
                   cv::Mat kernel, std::vector<cv::Vec4i> &lines) {
    // 获取图像的高度和宽度
    int height = frame.rows;
    int width = frame.cols;

    // 定义感兴趣区域ROI
    int x1 = width / 4;
    int y1 = height / 3.5;
    int x2 = 3 * width / 4;
    int y2 = 3 * height / 4;
    cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    // 高斯模糊
	cv::GaussianBlur(frame, frame, cv::Size(7, 7), 0);

    // 将图像从BGR转换为HSV格式
    cv::Mat hsv_image;
    cv::cvtColor(frame, hsv_image, cv::COLOR_BGR2HSV);

    // 创建左右颜色掩膜
    cv::Mat left_mask, right_mask;
    cv::inRange(hsv_image, left_lower_hsv, left_upper_hsv, left_mask);
    cv::inRange(hsv_image, right_lower_hsv, right_upper_hsv, right_mask);

    // 使用掩膜分割颜色区域
    cv::Mat left_result, right_result;
    cv::bitwise_and(frame, frame, left_result, left_mask);
    cv::bitwise_and(frame, frame, right_result, right_mask);

    // 合并左右分割结果
    cv::Mat combined_result;
    cv::addWeighted(left_result, 1, right_result, 1, 0, combined_result);

    // 去除噪点 - 应用开运算
    cv::morphologyEx(combined_result, combined_result, cv::MORPH_OPEN, kernel);

    // 二值化
    cv::threshold(combined_result, combined_result, 127, 255, cv::THRESH_BINARY);

    // 检测边缘
    cv::Mat edges;
    cv::Canny(roi, edges, 50, 150);

    // 霍夫变换检测直线
    cv::HoughLinesP(edges, lines, 1, CV_PI / 720, 130, 150, 200);
    // 在图像上绘制直线(size_t i = 0; i < lines.size(); i++)
    for (size_t i = 0; i <lines.size(); i++) {
        cv::line(roi, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Edges", edges);
    //cv::imshow("Combined", combined_result);
}

int main() {
    // 初始化摄像头
    cv::VideoCapture cap(0); // 0表示默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 设置串口
    uart_init();

    // 定义HSV颜色范围
    cv::Scalar left_lower_hsv(20, 10, 100);
    cv::Scalar left_upper_hsv(40, 50, 255);
    cv::Scalar right_lower_hsv(50, 10, 50);
    cv::Scalar right_upper_hsv(70, 50, 200);

    // 定义结构元素
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
    char flag = 0;

    // 主循环
    while (true) {
        cv::Mat frame;
        cap >> frame;  // 读取摄像头帧
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture frame\n";
            break;
        }

        std::vector<cv::Vec4i> lines; // 在每次循环中定义 lines 以避免未定义行为
        process_frame(frame, left_lower_hsv, left_upper_hsv, right_lower_hsv, right_upper_hsv, kernel, lines);
        
        // 显示结果
        cv::imshow("Detected Lines", frame);
        
        send_slope_data(lines, frame);
        
        if (serialDataAvail(fd)) {
            flag = serialGetchar(fd);
        }

        if (flag == 'a') {
            std::cout << "Received 'a', starting image processing...\n";
            if (!lines.empty()) { // 确保 lines 不为空
                send_slope_data(lines, frame);
                //print(lines);
                flag = 0;
            }
        }
		
        // 按下 'q' 键退出
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // 释放资源
    cap.release();
    serialClose(fd); // 使用 wiringPi 关闭串口
    cv::destroyAllWindows();

    return 0;
}
