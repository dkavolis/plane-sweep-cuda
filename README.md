# plane-sweep-cuda
**C++ plane sweep algorithm using CUDA**

Works with living room image set from [here](http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html). Download *ICL-NUIM PNGs*  
Depthmap fusion in progress.

**Not bug free.** [CMakeLists.txt](https://github.com/DKavolis/plane-sweep-cuda/blob/master/src/CMakeLists.txt) 
will most likely need to be configured for your system.

**Libraries required:**
* [Qt](http://www.qt.io/)
* [VTK](http://www.vtk.org/)
* [PCL](http://pointclouds.org/)
* [CUDA](https://developer.nvidia.com/cuda-zone)
* [boost](http://www.boost.org/)

**Optional libraries:**
* [OpenCV](http://opencv.org/)

There are 2 ways to create variables on managed memory, that inherit from class *Managed*:
```
  ExampleStruct * example = new ExampleStruct;
```
or
```
  ExampleStruct & example = *(new ExampleStruct);
```
