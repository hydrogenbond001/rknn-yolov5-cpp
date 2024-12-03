//this is for video detect and line_detected with usart communicatation
//you can switch them in cmakelist.txt 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <termios.h>
#include <sys/time.h>
#include <pthread.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <dlfcn.h>

// OpenCV 库
#include <opencv2/opencv.hpp>

// WiringPi 库，用于 GPIO 和串口通信
#include <wiringPi.h>
#include <wiringSerial.h>

// RKNN 和 RGA 库，使用于模型推理与图像处理
#include "RgaUtils.h"
#include "postprocess.h"
#include "rknn_api.h"
#include "preprocess.h"

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
    //printf("UART initialized successfully!\n");
}



int x,y;//the pos of object

// 发送整数的函数
void serialPutNumber(const int fd, int num) {
    char buffer[32];  // 定义足够大的缓冲区来存储数字字符串
    snprintf(buffer, sizeof(buffer), "%d", num);  // 将整数转换为字符串
    serialPuts(fd, buffer);  // 通过 serialPuts 发送字符串
}

// 发送浮点数的函数
void serialPutFloat(const int fd, float num) {
    char buffer[32];  // 定义缓冲区来存储浮点数字符串
    snprintf(buffer, sizeof(buffer), "%.2f", num);  // 将浮点数转换为字符串，保留两位小数
    serialPuts(fd, buffer);  // 通过 serialPuts 发送字符串
}

//below is line detected
void send_slope_data(std::vector<cv::Vec4i> lines, cv::Mat &roi) {
    if (!lines.empty()) {
        int index_of_lowest_line = 0; // 记录最下面直线的索引
        int max_y2 = lines[0][3]; // 初始化为第一条直线的y2值

        // 找到最下面的直线
        for (size_t i = 1; i < lines.size(); i++) {
            if (lines[i][3] > max_y2) {
                max_y2 = lines[i][1];
                index_of_lowest_line = i;
            }
        }

        // 获取最下面的直线的坐标
        int x1 = lines[index_of_lowest_line][0];
        int y1 = lines[index_of_lowest_line][1];
        int x2 = lines[index_of_lowest_line][2];
        int y2 = lines[index_of_lowest_line][3];

        // 计算斜率
        if (x2 - x1 != 0) {
            // 计算斜率（乘以100调整到较大的范围）
            float slope = 10 * (180.0 / CV_PI) * (10.0 * (y2 - y1) / (x2 - x1));
		if(slope<2000){
			// 计算中点
            int midpoint_x = (x1 + x2) / 2;
            int midpoint_y = (y1 + y2) / 2;

            // 打印中点和斜率信息
            printf("%d %d %.0f\n", midpoint_x, midpoint_y, slope);

            // 发送起始字节
            unsigned char start_byte = 0x07;
            write(fd, &start_byte, 1);
            usleep(1000);

            // 发送斜率数据
            char slope_buffer[32];
            char slope_buffer1[32];
            snprintf(slope_buffer, sizeof(slope_buffer), "%.0f    %d", slope,midpoint_y);
            //snprintf(slope_buffer, sizeof(slope_buffer), "%.0f    ", slope);
            //snprintf(slope_buffer1, sizeof(slope_buffer1), "%d", midpoint_y);
            write(fd, slope_buffer, strlen(slope_buffer));
            usleep(1000);
		  //write(fd, slope_buffer1, strlen(slope_buffer1));
            // 发送结束字节
            unsigned char end_byte = 0xFE;
            write(fd, &end_byte, 1);
            usleep(1000);
        }
		}	
            

        // 打印最下面直线的坐标到终端
        std::cout << "Lowest Line: (" 
                  << lines[index_of_lowest_line][0] << ", " << lines[index_of_lowest_line][1] << ") -> (" 
                  << lines[index_of_lowest_line][2] << ", " << lines[index_of_lowest_line][3] << ")" << std::endl;
    }
}


void process_frame(cv::Mat frame, cv::Scalar left_lower_hsv, cv::Scalar left_upper_hsv,
                   cv::Scalar right_lower_hsv, cv::Scalar right_upper_hsv,
                   cv::Mat kernel, std::vector<cv::Vec4i> &lines) {
    // 获取图像的高度和宽度
    int height = frame.rows;
    int width = frame.cols;

    // 定义感兴趣区域ROI
    int x1 = width / 3;
    int y1 = height / 8;
    int x2 = 3 * width / 4;
    int y2 = 3 * height / 4;
    cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
/*
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
    cv::Mat edges;
    // 检测边缘
    edges = cv2.Canny(blurred, 100, 150)
    cv::Canny(roi, edges, 90, 150);
*/


	cv::Mat gray;
	cv::cvtColor(roi,gray, cv::COLOR_BGR2GRAY);
	cv::Mat edges;
	//使用高斯滤波去噪
	cv::GaussianBlur(gray,edges, cv::Size(7, 5), 0);
	
	//边缘检测
	
	cv::Canny(edges,edges, 70, 150);
    

    // 霍夫变换检测直线
    cv::HoughLinesP(edges, lines, 1, CV_PI / 720, 160, 100, 60);
    // 在图像上绘制直线(size_t i = 0; i < lines.size(); i++)
    for (size_t i = 0; i <lines.size(); i++) {
        cv::line(roi, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 0), 2);
    }
    //cv::imshow("Edges", edges);
    //cv::imshow("Combined", combined_result);
}




// Function prototypes
static void dump_tensor_attr(rknn_tensor_attr *attr);
double __get_us(struct timeval t);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (int i = 1; i < attr->n_dims; ++i)
  {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
  unsigned char *data;
  int ret;

  data = NULL;

  if (NULL == fp)
  {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0)
  {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char *)malloc(sz);
  if (data == NULL)
  {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp;
  unsigned char *data;

  fp = fopen(filename, "rb");
  if (NULL == fp)
  {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
  FILE *fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++)
  {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}


//串口发送协议
int send_serial_data(int fd, const char *data)
{
    if (write(fd, data, strlen(data)) < 0) {
        perror("Failed to send serial data");
        return -1;
    }
    
    return 0;
}
// 定义 run_inference 函数
void run_inference(rknn_context ctx, rknn_input_output_num io_num, rknn_tensor_attr* output_attrs, cv::Mat& img, int width, int height, int channel, float box_conf_threshold, float nms_threshold, int fd,int flag) 
{
    // Convert to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize image if needed
    cv::Mat resized_img;
    cv::Size target_size(width, height);
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;
    float min_scale = std::min(scale_w, scale_h);
    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    letterbox(img, resized_img, pads, min_scale, target_size);

    // Set input data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = resized_img.data;

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);

    // Set inputs and run inference
    rknn_inputs_set(ctx, io_num.n_input, inputs);
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    int ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    //printf("Inference time: %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // Post-process
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    
    post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width, box_conf_threshold, nms_threshold, pads, min_scale, min_scale, out_zps, out_scales, &detect_result_group);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    // Draw results
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        char text[256];
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        //printf("Detected Object: %s\n", text);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        int x = (x1 + x2) / 2;
        int y = (y1 + y2) / 2;
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
        cv::putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

        // 打印矩形框坐标和中心坐标
        //printf("Bounding Box: left=%d, top=%d, right=%d, bottom=%d\n", x1, y1, x2, y2);
        //printf("Center: x=%d, y=%d\n\n", x, y);
        //printf("   ");
        if(y>60){
    if (strcmp(det_result->name, "1") == 0 && flag=='R') {//blue

        printf("1");printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte1 = 0x01;
            write(fd, &start_byte1, 1);

            // 
            char obj1[32];
            snprintf(obj1, sizeof(obj1), "%d   %d\n", x,y);
            write(fd, obj1, strlen(obj1));usleep(1000);

            // 发送结束字节
            unsigned char end_byte1 = 0xFE;
            write(fd, &end_byte1, 1);
            usleep(1000);
    } 
    else if (strcmp(det_result->name, "2") == 0 && flag=='R') {

        printf("2");printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte2 = 0x02;
            write(fd, &start_byte2, 1);usleep(1000);

            // 
            char obj2[32];
            snprintf(obj2, sizeof(obj2), "%d   %d\n", x,y);
            write(fd, obj2, strlen(obj2));

            // 发送结束字节
            unsigned char end_byte2 = 0xFE;
            write(fd, &end_byte2, 1);
            usleep(1000);
    }
    else if (strcmp(det_result->name, "3") == 0 && flag=='G') {//green

        printf("3");printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte3 = 0x03;
            write(fd, &start_byte3, 1);usleep(1000);

            // 
            char obj3[32];
            snprintf(obj3, sizeof(obj3), "%d   %d\n", x,y);
            write(fd, obj3, strlen(obj3));

            // 发送结束字节
            unsigned char end_byte3 = 0xFE;
            write(fd, &end_byte3, 1);
            usleep(1000);
    } 
    else if (strcmp(det_result->name, "4") == 0 && flag=='G') {

        //printf("4");
        printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte4 = 0x04;
            write(fd, &start_byte4, 1);usleep(1000);

            // 
            char obj4[32];
            snprintf(obj4, sizeof(obj4), "%d   %d\n", x,y);
            write(fd, obj4, strlen(obj4));

            // 发送结束字节
            unsigned char end_byte4 = 0xFE;
            write(fd, &end_byte4, 1);
            usleep(1000);
    }
    else if (strcmp(det_result->name, "5") == 0 && flag=='B') {//red

        printf("5");printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte5 = 0x05;
            write(fd, &start_byte5, 1);usleep(1000);

            // 
            char obj5[32];
            snprintf(obj5, sizeof(obj5), "%d   %d\n", x,y);
            write(fd, obj5, strlen(obj5));

            // 发送结束字节
            unsigned char end_byte5 = 0xFE;
            write(fd, &end_byte5, 1);
            usleep(1000);
    } 
    else if (strcmp(det_result->name, "6") == 0 && flag=='B') {

        printf("6");printf("Center: x=%d, y=%d\n\n", x, y);printf("Detected Object: %s\n", text);
        // 发送起始字节K
            unsigned char start_byte6 = 0x06;
            write(fd, &start_byte6, 1);usleep(1000);

            // 
            char obj6[32];
            snprintf(obj6, sizeof(obj6), "%d   %d\n", x,y);
            write(fd, obj6, strlen(obj6));usleep(1000);

            // 发送结束字节
            unsigned char end_byte6 = 0xFE;
            write(fd, &end_byte6, 1);
            usleep(1000);
    }
    }
    else{}//任意y大于多少个像素不处理
    }

    // Release RKNN outputs
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    // Free the resized image memory
    resized_img.release();
}




int main(int argc, char **argv)
{
    uart_init();
	//for object----------------------------------------------------------------------------------------------
    int ret;
    rknn_context ctx;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    

    const char *model_name = "/home/orangepi/rknn-cpp-Multithreading-main/model/RK3588/yolov5s.rknn";

    printf("Loading model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query error ret=%d\n", ret);
        return -1;
    }

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    //for object----------------------------------------------------------------------------------------------
    
    //for lines----------------------------------------------------------------------------------------------// 初始化摄像头

    
    cv::VideoCapture cap(0); // 0表示默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap1(2); // 0表示默认摄像头
    if (!cap1.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

	// 定义HSV颜色范围
    cv::Scalar left_lower_hsv(20, 10, 100);
    cv::Scalar left_upper_hsv(40, 50, 255);
    cv::Scalar right_lower_hsv(50, 10, 50);
    cv::Scalar right_upper_hsv(70, 50, 200);

    // 定义结构元素
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
    char flag = 0;



    
    // 主循环
    cv::Mat img;
    cv::Mat frame;
    while (true)
    {

        
        cap1 >> frame;  // 读取摄像头帧
        if (frame.empty()) {
            std::cerr << "Error: Unable to capture frame\n";
            break;
        }

        cap >> img;
        if (img.empty())
        {
            printf("Empty frame\n");
            break;
        }
        
        std::vector<cv::Vec4i> lines; // 在每次循环中定义 lines 以避免未定义行为
        process_frame(frame, left_lower_hsv, left_upper_hsv, right_lower_hsv, right_upper_hsv, kernel, lines);

        //run_inference(ctx, io_num, output_attrs, img, width, height, channel, box_conf_threshold, nms_threshold, fd,flag);
	   //usleep(100000);
        //send_slope_data(lines, frame);
        //usleep(10000);
        if (serialDataAvail(fd)) {
            flag = serialGetchar(fd);
        }
        //flag='G';
	   
        if (flag == 'a') {//for line
            std::cout << "Received 'a', starting image processing...\n";
            if (!lines.empty()) { // 确保 lines 不为空
                send_slope_data(lines, frame);
            }
        }
        run_inference(ctx, io_num, output_attrs, img, width, height, channel, box_conf_threshold, nms_threshold, fd,flag);
        
        if(flag == 'c'){
        	flag = 0;
        }
	   // 显示结果
        cv::imshow("Lines", frame);
        cv::imshow("img", img);
        // 按下 'q' 键退出
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // Cleanup
    cap.release();
    rknn_destroy(ctx);
    cv::destroyAllWindows();
    free(model_data);
    deinitPostProcess(); // Release post-process resources

    serialClose(fd); // 使用 wiringPi 关闭串口

    return 0;
}
