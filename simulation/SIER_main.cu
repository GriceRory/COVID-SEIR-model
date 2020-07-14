#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

#include "simulate.cu"


void printSimulationLayer(double* layer) {
    printf("susceptible, exposed, presymptomatic, infectiousUntested, infectiousTested, recoveredUntested, recoveredTested, Deaths\n");
    //printf("beta, epsilon, alpha, delta, gamma, testingRate, population, ICUbeds\n");
    for (int i = 0; i < 7; ++i) {
        printf("%f, ", layer[i] * 5000000);
    }
    printf("%f\n\n", layer[7] * 5000000);
}

void initializeConstants(double *constant, double *input) {
    alpha = 0.25;
    delta = 1;
    gamma = 0.1;
    epsilon = 0.15;
    double R0 = 2.5;
    beta = R0 / (epsilon / delta + 1 / gamma);
    testingRate = 1.0;

    population = 5000000;
    exposed = 10/population;
    presymptomatic = 0;
    ICUbeds = 500/population;
    infectiousUntested = 0;
    infectiousTested = 0;
    recoveredUntested = 0;
    recoveredTested = 0;
    deaths = 0;
    susceptible = 1 - exposed - presymptomatic - infectiousTested - infectiousUntested - recoveredTested - recoveredUntested - deaths;
}

int main() {
    double* h_constant;
    double* h_input;
    cudaMallocHost(&h_constant, 8 * sizeof(double));
    cudaMallocHost(&h_input, 8 * sizeof(double));
    
    initializeConstants(h_constant, h_input);
    
    double* d_constant;
    double* d_input;
    cudaMalloc(&d_constant, 8 * sizeof(double));
    cudaMalloc(&d_input, 8 * sizeof(double));
    cudaMemcpy(d_constant, h_constant, 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, 8 * sizeof(double), cudaMemcpyHostToDevice);
    
    double simulationStepSize = 0.5;
    int simulationSteps = 365/simulationStepSize;
    
    double** h_simulation = simulate(d_input, d_constant, simulationSteps, simulationStepSize);
    cudaDeviceSynchronize();
    int error = cudaGetLastError();

    for (int i = 0; i < simulationSteps; ++i) {
        printSimulationLayer(h_simulation[i]);
    }



    printf("finished with cuda error %d", error);
    return error;
}
