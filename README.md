# 简介
* 此仓库为c++实现, 大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2)



# 使用说明
  注意要在CMakeLists.txt里把add_executable(main ...）改掉，直接运行我的程序是不行的，我的程序有两个摄像头同时还用了串口，需要对cmake有点了解，有空再整理
  ```
  bash ./build-linux_RK3588.sh 
  cd install/rknn_yolov5_demo_Linux/
  ./main
  ```
### 演示
  * 系统需安装有**OpenCV**
  * 下载Releases中的测试视频于项目根目录,运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo
  * 终端参数输入命令写死在程序里了

### 部署应用
  * 

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)


# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo
* https://github.com/leafqycc/rknn-cpp-Multithreading