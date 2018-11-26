#include <iostream>
#include <random>
#include "Convolution.h"

//! Creates and returns a random vector given the size.
//! Values (-5, 5).
//! \param size the desired size of the return vector.
//! \return the random vector
std::vector<double> makeRandomVector(long size) {
    std::vector<double> out;
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    for (long i = 0; i < size; i++)
        out.push_back(unif(re));

    return out;
}

//! Creates and returns a random 2-dimensional vector given the size.
//! \param rows how many rows is needed
//! \param columns how many columns is needed
//! \return the random vector
std::vector<std::vector<double>> makeRandom2DVector(long rows, long columns) {
    std::vector<std::vector<double>> out;
    std::uniform_real_distribution<double> unif(-5.0, 5.0);
    std::default_random_engine re;

    for (long i = 0; i < rows; i++) {
        std::vector<double> row;
        for (long j = 0; j < columns; j++)
            row.push_back(unif(re));

        out.push_back(row);
    }

    return out;
}

//! Creates and returns a single value vector given the size.
//! \param size the desired size of the return vector.
//! \param value the desired value
//! \return the random vector
std::vector<double> makeValueVector(long size, double value) {
    std::vector<double> out;

    for (long i = 0; i < size; i++)
        out.push_back(value);

    return out;
}

//! Creates and returns a 2-dimensional single value vector given the size.
//! \param rows how many rows is needed
//! \param columns how many columns is needed
//! \param value the desired value
//! \return the random vector
std::vector<std::vector<double>> makeValue2DVector(long rows, long columns, double value) {
    std::vector<std::vector<double>> out;

    for (long i = 0; i < rows; i++) {
        std::vector<double> row;
        for (long j = 0; j < columns; j++)
            row.push_back(value);

        out.push_back(row);
    }

    return out;
}

void runPredefinedTests() {
    std::vector<double> input_1d;
    std::vector<double> filter_1d;

    unsigned long size = 0;

    std::cout << "please type the size of the input: ";
    std::cin >> size;
    input_1d = makeRandomVector(size);
    filter_1d = makeValueVector(5, 1.0 / 5);

    std::vector<double> out = Convolution::myConvolve(input_1d, filter_1d);
    std::cout << std::endl;
    for (double &i : out)
        std::cout << i << " ";

    /* --------------------------------------------------------------*/
    std::cout << std::endl;

    std::vector<std::vector<double>> input_2d;
    std::vector<std::vector<double>> filter_2d;
    unsigned long rows = 0, columns = 0;

    std::cout << "please type the rows of the input: ";
    std::cin >> rows;
    std::cout << "please type the columns of the input: ";
    std::cin >> columns;
    input_2d = makeRandom2DVector(rows, columns);
    filter_2d = makeValue2DVector(3, 3, 1.0 / 9);

    std::vector<std::vector<double>> conv = Convolution::myConvolve(input_2d, filter_2d);
    std::cout << std::endl;
    for (std::vector<double> &r : conv) {
        for (double &i : r)
            std::cout << i << " ";

        std::cout << std::endl;
    }
}

int main() {

    // run the tests
    runPredefinedTests();

    return 0;
}

