# Object_detection
What is this ? 
--------------
Object detection application. Demo_app uses two input images. It matches features in one image with features from other image and finds Homography from calib3d module. Finally finds-overlay known objects. 

Current imp. includes two algorithms (SIFT/SURF data-parallel(OpenCL), optimized for GPU/CPU) implemented with coopcl driver that enables cooperative, async. and parallel CPU/GPU execution. 

Requierments ?
---------------
1. C++14 compiler 
2. CMake 3.x
3. OpenCL 2.x headers and lib, support for CPU and GPU
3. GPU driver with OpenCL and SVM_FINE_GRAIN_BUFFER support
4. For unit-tests CTest
5. OpenCV 4.x (modules calib3d,GUI,image,core)

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

References
------------
1)  A framework for accelerating local feature extraction with OpenCL on multi-core CPUs and co-processors. Journal of real-time image processing. K.Moren and D.Göhringer https://dl.acm.org/doi/abs/10.1007/s11554-016-0576-0

2) clSURF https://github.com/perhaad/clsurf/tree/master/CLSource 
