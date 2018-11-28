#include <iostream>
#include <random>

__global__
void conv1d(int input_size, int filter_size, double *input, double *filter, double *conv)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < input_size; i+= stride)
        for(int j = 0, j_end = (i < filter_size) ? i + 1 : filter_size; j<j_end; j++)
            conv[i]+=input[i-j]*filter[j];
}

int main() {
    // this will be the 1d input vector
    int input_size;
    std::cout << "please type the size of the input: ";
    std::cin>>input_size;

    double *input, *filter, *conv;

    // shared mem allocation for cpu-gpu
    cudaMallocManaged(&input, input_size*sizeof(double));
    cudaMallocManaged(&conv, input_size*sizeof(double));
    cudaMallocManaged(&filter, 5*sizeof(double));
    
    // 
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    // fill the input vector with random values
    for(int i = 0; i < input_size; i++)
    {
        input[i] = unif(re);
        conv[i] = 0;
    }
        

    for(int i = 0; i<5; i++)
        filter[i] = 1.0/5;

    conv1d<<<1, 256>>>(input_size, 5, input, filter, conv);

    // wait for the gpu thread to end
    cudaDeviceSynchronize();

    std::cout << "RESULTS: ";
    for(int i = 0; i < input_size; i++)
    {
        std::cout << conv[i] << " ";
    }
    std::cout << std::endl;

    // free the input & filter mem
    cudaFree(input);
    cudaFree(filter);
    cudaFree(conv);

    return 0;
}
