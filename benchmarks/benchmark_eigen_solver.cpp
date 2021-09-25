/**
 * @file benchmark_eigen_solver.cpp
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-09-25
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <benchmark/benchmark.h>

#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Sparse>

//  SIZE_SYSTEM is basically the number of nodes in the mesh
static constexpr std::size_t SIZE_SYSTEM = 10;
//  NUMBER_ITERATION is used to simulate the great number of time poisson equation will be solved in MonteCarlo process.
static constexpr std::size_t NUMBER_ITERATION = 10'000;

    Eigen::SparseMatrix<double>
    create_laplacian_matrix(const std::size_t size) {
    typedef Eigen::Triplet<double> T;
    std::vector<T>                 tripletList;
    const std::size_t              non_zero_estimation = 3 * size;
    tripletList.reserve(non_zero_estimation);
    //  Filling respectively the upper diagonal, the diagonal and the lower diagonal.
    for (std::size_t index_row = 1; index_row < size - 1; ++index_row) {
        tripletList.push_back(T(index_row, index_row + 1, -1.0));
        tripletList.push_back(T(index_row, index_row, 2.0));
        tripletList.push_back(T(index_row - 1, index_row, -1.0));
    }
    //  Filling the first and last line of the matrix
    tripletList.push_back(T(0, 0, 2.0));
    tripletList.push_back(T(size - 1, size - 1, 2.0));

    Eigen::SparseMatrix<double> MatrixPoisson(size, size);
    MatrixPoisson.setFromTriplets(tripletList.begin(), tripletList.end());

    std::cout << MatrixPoisson << std::endl;
    return MatrixPoisson;
}

Eigen::VectorXd create_random_vector(const std::size_t size) {
    Eigen::VectorXd RandomVector = Eigen::VectorXd::Random(size);
    return RandomVector;
}

static void EIGEN_LU_BENCH(benchmark::State &state) {
    Eigen::SparseMatrix<double> PoissonMatrix = create_laplacian_matrix(SIZE_SYSTEM);
    Eigen::VectorXd RandomVector = create_random_vector(SIZE_SYSTEM);

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > LU_Solver;
    LU_Solver.analyzePattern(PoissonMatrix);
    LU_Solver.factorize(PoissonMatrix);
    for (auto _ : state) {
        for (std::size_t iteration = 0; iteration < NUMBER_ITERATION; ++ iteration) {
            LU_Solver.solve(RandomVector);
        }
    }
}

BENCHMARK(EIGEN_LU_BENCH);
// Run the benchmark
BENCHMARK_MAIN();