# Simple Convolution
> This is a simple convolution implementation both for CPU_only and GPU_only (using CUDA).

In this project I implimented the convolution algorithm both for a 1d vector and a 2d vector. Both implimentations are templated so you can pass any typename you need.

## Getting started
Firstly, iclude the _Convolution.h_ file.
`#include "Convolution.h"`

Then you can use the static methods _myConvolve_ passing the correct inputs 

### 1D vectors
To use the one dimentional convolution function just call the _myConvolve_ function with 2 vectors of doubles.
```
std::vector<T> input, filter, conv;
conv = nvolution::myConvolve(input, filter);
```

<img width="400" height="340" alt="1d-conv" src="https://i.stack.imgur.com/kTBiy.gif">

### 2D vectors
To use the 2 dimentional convolution function just call the _myConvolve_ function with 2 2d vectors of doubles.
```
std::vector<std::vector<T>> input, filter, conv;
conv = nvolution::myConvolve(input, filter);
```

<img width="600" height="340" alt="1d-conv" src="https://cdn-images-1.medium.com/max/1600/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif">
