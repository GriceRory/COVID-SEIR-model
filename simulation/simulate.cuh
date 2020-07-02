
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "rungeKutta4.cuh"

double** simulate(double* inputs, double* constants, int steps, double stepSize);
double getR0(double* constants);

/*
int main(void)
{
    


    double** device_simulation = simulate(input, constant, simulationSteps, simulationStepSize);

    double** host_simulation = (double**)malloc(sizeof(double*) * simulationSteps);
    for (int i = 0; i < simulationSteps; ++i) {
        host_simulation[i] = (double*)malloc(sizeof(double) * 8);
        cudaMemcpy(host_simulation[i], device_simulation[i], sizeof(double) * 8, cudaMemcpyDeviceToHost);
    }

    return 0;
}*/

double** simulate(double* inputs, double* constants, int steps, double stepSize) {




    double** simulationResults;
    cudaMalloc(&simulationResults, sizeof(double*) * steps);
    for (int i = 0; i < steps; ++i) {
        cudaMalloc(&simulationResults[i], sizeof(double) * 8);
    }


    rungaKutta4<<<1, 8>>>(inputs, constants, steps, stepSize, simulationResults);
    return simulationResults;
}


double getR0(double* constant) {
    return beta * (epsilon / delta + 1 / gamma);
}





