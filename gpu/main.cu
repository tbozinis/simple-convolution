#include <iostream>
#include <string>
#include <random>
#include <fstream>

//! One dimensional convolution.
//! \param input_size input vector size
//! \param filter_size filter vector size
//! \param input the input vector
//! \param filter the filter vector
//! \param conv the convolution vector
__global__
void myConvolution(int input_size, int filter_size, double *input, double *filter, double *conv) {
    int index = threadIdx.x;
    int stride = blockDim.x;
    // iterate through the first vector and foreach cell run the filter
    for (int i = index; i < input_size; i += stride)
        for (int j = 0, j_end = (i < filter_size) ? i + 1 : filter_size; j < j_end; j++)
            conv[i] += input[i - j] * filter[j];
}

//! Two dimensional convolution.
//! \param input_size_n #rows of input array
//! \param input_size_m #columns of input array
//! \param filter_size_n #rows of filter array
//! \param filter_size_m #columns of filter array
//! \param input the input array
//! \param filter the filter array
//! \param conv the convolution array
__global__
void myConvolution(int input_size_n, int input_size_m, int filter_size_n, int filter_size_m,
                   double **input, double **filter, double **conv) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    // iterate through the first vector
    for (int row = index; row < input_size_n; row += stride) {
        for (int column = 0; column < input_size_m; column++) {
            // if the filter doesn't fit continueinput
            if (filter_size_n + row > input_size_n || filter_size_m + column > input_size_m)
                continue;

            double v = 0;

            // run the filter on the input vector
            for (int i = 0; i < filter_size_n; i++)
                for (int j = 0; j < filter_size_m; j++)
                    v += input[row + i][column + j] * filter[i][j];

            conv[row][column] = v;
        }
    }
}

//! Prints the results to a file named "results 1d.txt"
//! \param input the input array
//! \param input_n #size of input array
//! \param filter the filter array
//! \param filter_n #size of filter array
//! \param conv the convolution array
//! \param conv_n #size of convolution array
void printFiles(double *input, int input_n,
                double *filter, int filter_n,
                double *conv, int conv_n) {
    std::ofstream out;
    out.open("results 1d.txt");
    out << "These are the results of the 1d convolution of the input and filter that are shown below." << std::endl
        << std::endl;

    out << "INPUT VECTOR: ";
    for (int i = 0; i < input_n; i++)
        out << input[i] << " ";
    out << std::endl << std::endl;

    out << "FILTER VECTOR: ";
    for (int i = 0; i < filter_n; i++)
        out << filter[i] << " ";
    out << std::endl << std::endl;

    out << "CONVOLUTION (time = " << time << " ms) (input*filter) VECTOR: ";
    for (int i = 0; i < conv_n; i++)
        out << conv[i] << " ";
}

//! Prints the results to a file named "results 2d.txt"
//! \param input the input array
//! \param input_n #rows of input array
//! \param input_m #columns of input array
//! \param filter the filter array
//! \param filter_n #rows of filter array
//! \param filter_m #columns of filter array
//! \param conv the convolution array
//! \param conv_n #rows of convolution array
//! \param conv_m #columns of convolution array
void
printFiles2d(double **input, int input_n, int input_m,
             double **filter, int filter_n, int filter_m,
             double **conv, int conv_n, int conv_m) {

    std::ofstream out;
    out.open("results 2d.txt");
    out << "These are the results of the 1d convolution of the input and filter that are shown below." << std::endl
        << std::endl;
    out << "INPUT ARRAY: " << std::endl;
    for (int i = 0; i < input_n; i++) {
        for (int j = 0; j < input_m; j++)
            out << input[i][j] << " ";

        out << std::endl;
    }

    out << std::endl;
    out << "FILTER ARRAY: " << std::endl;
    for (int i = 0; i < filter_n; i++) {
        for (int j = 0; j < filter_m; j++)
            out << filter[i][j] << " ";

        out << std::endl;
    }
    out << std::endl;

    out << "CONVOLUTION (input*filter) ARRAY: " << std::endl;
    for (int i = 0; i < conv_n; i++) {
        for (int j = 0; j < conv_m; j++)
            out << conv[i][j] << " ";

        out << std::endl;
    }
}

int main() {

    std::cout << std::endl << " ------------------------- 1D convolution using CUDA ------------------------- "
              << std::endl << std::endl;

    // this will be the 1d input vector
    int input_size;
    std::cout << "please type the size of the input: ";
    std::cin >> input_size;

    double *input, *filter, *conv;

    // shared mem allocation for cpu-gpu
    cudaMallocManaged(&input, input_size * sizeof(double));
    cudaMallocManaged(&conv, input_size * sizeof(double));
    cudaMallocManaged(&filter, 5 * sizeof(double));

    // initialize a uniform distribution for our input vector
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    // fill the input vector with random values
    for (int i = 0; i < input_size; i++) {
        input[i] = unif(re);

        // fill the conv array with 0s
        conv[i] = 0;
    }

    // fill the filter with 1/5
    for (int i = 0; i < 5; i++)
        filter[i] = 1.0 / 5;

    myConvolution << < 1, 256 >> > (input_size, 5, input, filter, conv);

    // wait for the gpu thread to end
    cudaDeviceSynchronize();

    // print the results
    std::string answer;
    std::cout << "Do you want to print the results? (yes, no): ";
    std::cin >> answer;
    if (answer == "yes") {
        std::cout
                << "==================================================================================================="
                << std::endl << std::endl;
        std::cout << "RESULTS: ";
        for (int i = 0; i < input_size; i++) {
            std::cout << conv[i] << " ";
        }
        std::cout << std::endl
                  << "==================================================================================================="
                  << std::endl << std::endl;
    }

    // print the results in a file
    printFiles(input, input_size, filter, 5, conv, input_size);

    // free the input & filter and the conv vectors
    cudaFree(input);
    cudaFree(filter);
    cudaFree(conv);

    std::cout << std::endl << " ------------------------- 2D convolution using CUDA ------------------------- "
              << std::endl << std::endl;

    double **input_2d, **filter_2d, **conv_2d;
    int rows = 0, columns = 0;

    std::cout << "please type the rows of the input: ";
    std::cin >> rows;
    std::cout << "please type the columns of the input: ";
    std::cin >> columns;

    cudaMallocManaged(&input_2d, rows * sizeof(double *));
    cudaMallocManaged(&filter_2d, 3 * sizeof(double *));
    // we allocate (rows - filter rows + 1) rows for the convolution array
    // in this case our filter is a 3x3 array so we allocate (rows - 2)
    cudaMallocManaged(&conv_2d, (rows - 2) * sizeof(double *));

    // init the input array and the conv array
    for (int i = 0; i < rows; i++) {
        // same rouls aplies here as well 
        cudaMallocManaged(&input_2d[i], columns * sizeof(double));
        cudaMallocManaged(&conv_2d[i], (columns - 2) * sizeof(double));
        for (int j = 0; j < columns; j++) {

            input_2d[i][j] = unif(re);

            // fill the conv array with 0s
            if (j < columns - 2)
                conv_2d[i][j] = 0;
        }
    }

    // init the filter
    for (int i = 0; i < 3; i++) {
        cudaMallocManaged(&filter_2d[i], 3 * sizeof(double));
        for (int j = 0; j < 3; j++)
            filter_2d[i][j] = 1.0 / 9;
    }

    myConvolution << < 1, 256 >> > (rows, columns, 3, 3, input_2d, filter_2d, conv_2d);

    // wait for the gpu thread to end
    cudaDeviceSynchronize();

    // print the results
    std::cout << "Do you want to print the results? (yes, no): ";
    std::cin >> answer;
    if (answer == "yes") {
        std::cout
                << "==================================================================================================="
                << std::endl << std::endl;
        std::cout << "RESULTS:" << std::endl;
        for (int i = 0; i < rows - 2; i++) {
            for (int j = 0; j < columns - 2; j++)
                std::cout << conv_2d[i][j] << " ";

            std::cout << std::endl;
        }
        std::cout << std::endl
                  << "==================================================================================================="
                  << std::endl << std::endl;
    }

    // print the results in a file
    printFiles2d(input_2d, rows, columns, filter_2d, 3, 3, conv_2d, rows - 2, columns - 2);

    // free the input & filter and conv arrays
    cudaFree(input_2d);
    cudaFree(filter_2d);
    cudaFree(conv_2d);

    return 0;
}
