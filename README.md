# 简介
* 此仓库为c++实现，较于python效率更高，环境简单
* 本程序在板端执行进行编译，编译完就可以直接运行，无需交叉编译和繁琐的文件互传
* 推荐去[这里](https://github.com/hydrogenbond001/atom_rk3568_yolov5)可以在两个平台使用
# 使用说明
  直接运行我的程序是不行的，我的程序用了两个摄像头同时还用了串口，需要对cmake有点了解，有空再整理
  注意要在CMakeLists.txt里把add_executable(main ...）改掉，new.cc是视频推理部分，pic是图片推理。
  ```
  bash ./build-RK3588.sh 
  cd install/rknn_yolov5_demo_Linux/
  ./main
  ```
### 演示
  * 系统需安装有**OpenCV**和CMake，Orangepi5_1.1.8_ubuntu_jammy_desktop_xfce_linux6.1.43里已经装了opencv4.5.4
  * 需要在orangepi-config中打开硬件上的串口0来进行通讯（uart0-m2）懒得开启也可以把new.cc里的串口部分删去
  * 运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo
  * 终端参数输入命令写死在程序里了，我只是将官方的命令简单的注释了，略微修改即可在终端中输入

### 部署应用
  * 

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)


# Acknowledgements
* https://github.com/leafqycc/rknn-cpp-Multithreading
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
