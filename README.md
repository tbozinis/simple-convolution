# Simple Convolution
> This is a simple convolution implementation both for CPU_only and GPU_only (using CUDA) 

In this project I implimented the convolution algorithm both for a 1d vector and a 2d vector.

## Getting started
Firstly, iclude the _Convolution.h_ file.
`#include "Convolution.h"`

Then you can use the static methods _myConvolve_ passing the correct inputs 

### 1D vectors
To use the one dimentional convolution function just call the _myConvolve_ function with 2 vectors of doubles.
```
std::vector<double> input, filter, conv;
conv = nvolution::myConvolve(input, filter);
```

[![gif](https://i.stack.imgur.com/kTBiy.gif)]()

### 2D vectors
To use the 2 dimentional convolution function just call the _myConvolve_ function with 2 2d vectors of doubles.
```
std::vector<std::vector<double>> input, filter, conv;
conv = nvolution::myConvolve(input, filter);
```

[![gif](https://cdn-images-1.medium.com/max/1600/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)]()
