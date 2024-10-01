## Readme For yolov8 camera stream using cpp


### Prerequisites
- Tested and working on Ubuntu 22.04
- Install CUDA, instructions [here](https://developer.nvidia.com/cuda-downloads).
  - Here i used version 12.4
- Install cudnn, instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download).
  - Here i used version 8

- #### Source buil OpenCV with cuda support.
  - version 4.10.0 + opencv_contrib4.x
  - use build command 

    `cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN=<target_architecture> -D CUDA_ARCH_PTX=<target_architecture> -D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -D WITH_CUBLAS=1 ..`

   - <target_architecture>  is a sm , Mine is gtx 1650 card so sm is 7.5, so i use CUDA_ARCH_BIN=7.5
   - <opencv_contrib> is a path of the opencv_contrib folder.

    - #### After cmake completed : 
        - make
        - suda make install

#### Build this yolo_stream_cpp:

    cd cpp_stream_yolo8
    mkdir build & cd build
    cmake .. && make

#### Run yolo_stream_cpp:

    ./yolo_vstream

   

