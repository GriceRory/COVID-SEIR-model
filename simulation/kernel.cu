
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <stdio.h>



#define beta constant[0]
#define epsilon constant[1]
#define alpha constant[2]
#define delta constant[3]
#define gamma constant[4]
#define testingRate constant[5]
#define population constant[6]
#define ICUbeds constant[7]

#define susceptible input[0]
#define exposed input[1]
#define presymptomatic input[2]
#define infectiousUntested input[3]
#define infectiousTested input[4]
#define recoveredUntested input[5]
#define recoveredTested input[6]
#define deaths input[7]



__global__ void rungeKutta4(double* inputs, double* constants, int steps, double stepSize, double** simulation) {
    //&inputs for each layer = &yValues[0]
    //&y0+h*k1/2 = &yValues[8]
    //&y0+h*k2/2 = &yValues[16]
    //&y0+h*k3 = &yValues[24]
    //&constants = &yValues[32]
    __shared__ double yValues[8 * 5];
    yValues[threadIdx.x] = inputs[threadIdx.x];
    yValues[threadIdx.x + 8 * 4] = constants[threadIdx.x];
    simulation[0][threadIdx.x] = inputs[threadIdx.x];

    double* constant = &yValues[8 * 4];
    double* input = yValues;

    /*
    //this lambda function would be nice to use because it cleans things up, but it doesnt want to work for some reason.
    //once I figure out that reason I might figure out a way to make it work with the lambda function and see if I can put it back in
    
    auto PartialDifferentialEquation = [] __device__(double* input, double* constant, int DifferentialEquation) {
        double IFR = 0.01;
        if (constant[6] * input[4] * 0.0125 > constant[7]) { 
            IFR = 0.02 - 0.01 * constant[7] / (constant[6] * input[4] * 0.0125);
        }
        switch (DifferentialEquation) {
        case 0:
            //  dSuseptible/dt
            return (-beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested));
        case 1:
            //  dExposed/dt
            return (beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested) - alpha * exposed);
        case 2:
            //  dPresymptomatic/dt
            return (alpha * exposed - delta * presymptomatic);
        case 3:
            //  dInfectedUntested/dt
            return (delta * presymptomatic - (gamma + testingRate) * infectiousUntested);
        case 4:
            //  dInfectedTested/dt
            return (testingRate * infectiousUntested - gamma * infectiousTested);
        case 5:
            //  dRecoveredUntested/dt
            return (gamma * infectiousUntested * (1 - IFR));
        case 6:
            //  dRecoveredTested/dt
            return (gamma * infectiousTested * (1 - IFR));
        case 7:
            //  Dead
            return (1 - susceptible - exposed - presymptomatic - infectiousTested - infectiousUntested - recoveredTested - recoveredUntested);
        }
        return 0.0;
    };*/
                            
    //this is an intellisense bug, the compiler handles it just fine.
    __syncthreads();

    double k[4];
    if (threadIdx.x != 0) { return; }
    for (int i = 1; i < steps; ++i) {
        for (int j = 0; j < 3; ++j) {

            input = &yValues[8 * j];
            double IFR = 0.01;
            if (constants[6] * inputs[4] * 0.0125 > constants[7]) {
                IFR = 0.02 - 0.01 * constants[7] / (constants[6] * inputs[4] * 0.0125);
            }
            switch (threadIdx.x) {
            case 0:
                //  dSuseptible/dt
                k[j] = stepSize * (-beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested));
                break;
            case 1:
                //  dExposed/dt
                k[j] = stepSize * (beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested) - alpha * exposed);
                break;
            case 2:
                //  dPresymptomatic/dt
                k[j] = stepSize * (alpha * exposed - delta * presymptomatic);
                break;
            case 3:
                //  dInfectedUntested/dt
                k[j] = stepSize * (delta * presymptomatic - (gamma + testingRate) * infectiousUntested);
                break;
            case 4:
                //  dInfectedTested/dt
                k[j] = stepSize * (testingRate * infectiousUntested - gamma * infectiousTested);
                break;
            case 5:
                //  dRecoveredUntested/dt
                k[j] = stepSize * (gamma * infectiousUntested * (1 - IFR));
                break;
            case 6:
                //  dRecoveredTested/dt
                k[j] = stepSize * (gamma * infectiousTested * (1 - IFR));
                break;
            case 7:
                //  Dead
                k[j] = stepSize * (1 - susceptible - exposed - presymptomatic - infectiousTested - infectiousUntested - recoveredTested - recoveredUntested);
                break;
            }

            if (j < 2) { k[j] /= 2; }
            yValues[8*(1+j) + threadIdx.x] = yValues[8*j + threadIdx.x] + k[j] / 2;
            __syncthreads();
        }

        simulation[i][threadIdx.x] = simulation[i - 1][threadIdx.x] + (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6;
        yValues[threadIdx.x] = simulation[i][threadIdx.x];
        __syncthreads();
    }
}

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
    testingRate = 0.1;

    population = 5000000;
    susceptible = 10/population;
    exposed = 20/population;
    presymptomatic = 0;
    ICUbeds = 500/population;
    infectiousUntested = 0;
    infectiousTested = 0;
    recoveredUntested = 0;
    recoveredTested = 0;
    deaths = 0;
}

double** simulate(double* inputs, double* constants, int steps, double stepSize) {
    double** h_simulation;
    double** d_simulation;
    cudaMallocHost(&h_simulation, steps * sizeof(double*), 0U);
    cudaMallocHost(&d_simulation, steps * sizeof(double*));

    for (int i = 0; i < steps; ++i) {
        double* h_temp;
        double* d_temp;
        cudaMallocHost(&h_temp, 8 * sizeof(double), 0U);
        cudaMalloc(&d_temp, 8 * sizeof(double));

        h_simulation[i] = h_temp;
        d_simulation[i] = d_temp;
    }
    
    rungeKutta4<<<1, 8>>>(inputs, constants, steps, stepSize, d_simulation);

    for (int i = 0; i < steps; ++i) { cudaMemcpy(h_simulation[i], d_simulation[i], 8 * sizeof(double), cudaMemcpyDeviceToHost); }
    return h_simulation;
}

int main() {
    double* h_constant;
    double* h_input;
    cudaMallocHost(&h_constant, 8 * sizeof(double));
    cudaMallocHost(&h_input, 8 * sizeof(double));
    
    initializeConstants(h_constant, h_input);

    printSimulationLayer(h_constant);
    printSimulationLayer(h_input);
    printf("done initializing variables\n\n\n\n\n\n\n\n\n\n\n");


    double* d_constant;
    double* d_input;
    cudaMalloc(&d_constant, 8 * sizeof(double));
    cudaMalloc(&d_input, 8 * sizeof(double));
    cudaMemcpy(d_constant, h_constant, 8 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, 8 * sizeof(double), cudaMemcpyHostToDevice);

    int simulationSteps = 2 * 10;
    double simulationStepSize = 0.5;

    double** h_simulation = simulate(d_input, d_constant, simulationSteps, simulationStepSize);
    cudaDeviceSynchronize();
    int error = cudaGetLastError();

    for (int i = 0; i < simulationSteps; ++i) {
        printSimulationLayer(h_simulation[i]);
    }

    printf("no runtime errors!!! cuda error %d", error);
    return error;
}
