#include <iostream>
#include <random>
#include "Convolution.h"

std::vector<double> makeVector(long size) {
    std::vector<double> out;
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    for (long i = 0; i < size; i++)
        out.push_back(unif(re));

    return out;
}

int main() {

    std::vector<double> temp1;
    std::vector<double> temp2;

    unsigned long size = 0;

    std::cout << "please type the size of the input: ";
    std::cin >> size;
    temp1 = makeVector(size);
    std::cout << std::endl << "please type the size of the filter: ";
    std::cin >> size;
    temp2 = makeVector(size);

    std::vector<double> out = Convolution::myConvolve(temp1, temp2);
    std::cout << std::endl;
    for (double &i : out)
        std::cout << i << ", ";

    return 0;
}

