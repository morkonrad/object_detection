# Object_detection
What is this ? 
--------------
Object detection application with SIFT/SURF algorithms. Demo_app uses two images, match features in one image with features from other and findHomography from calib3d module to find known objects in a complex image. 

Current imp. uses two algorithms (SIFT/SURF) implemented with coopcl driver that enables async. and parallel CPU/GPU execution. 

Requierments ?
---------------
1. C++14 compiler 
2. CMake 3.x
3. OpenCL 2.x headers and lib, support for CPU and GPU
3. GPU driver with OpenCL and SVM_FINE_GRAIN_BUFFER support
4. For unit-tests CTest
5. OpenCV (modules calib3d,GUI,image,core)

How to build ?
---------------
  1. git clone https://github.com/morkonrad/object_detection/ /dst
  2. cd /dst
  3. mkdir build 
  4. cd build
  5. cmake -G"Visual Studio 14 2015 Win64" -DOpenCV_DIR=your_path/opencv/build ..
  6. cmake --build . --config Release

How to use it ?
----------------
smaple usage:

cd build/SIFT/Release

./demo_coopcl_sift.exe -m ../../../data/matches/box.pgm ../../../data/matches/scene.pgm -f 0 "executes on CPU only"

./demo_coopcl_sift.exe -m ../../../data/matches/box.pgm ../../../data/matches/scene.pgm -f 1 "executes on GPU only"

./demo_coopcl_sift.exe -m ../../../data/matches/box.pgm ../../../data/matches/scene.pgm -f 0.5 "executes on CPU and GPU 50% CPU and 50% GPU"

./demo_coopcl_sift.exe -v ../../../data/basketball.mp4 -f 0.5 "executes on CPU and GPU 50% CPU and 50% GPU"

for more options call: ./demo_coopcl_sift.exe -h

