#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/detail/config.h>

#include <cuda.h>
#include <cuda_runtime.h>


//for __syncthreads() intellisense
#ifndef __CUDACC__ 
#define __CUDACC__
#endif



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
#define time input[8]

#define InfectionFatalityRate double IFR = 0.0;if(constant[6]*input[4]*0.0125 < constant[7]){IFR = 0.01;}else{IFR =  0.02 - constant[7]/(constant[6]*input[4]*0.0125)*(0.01);}


//takes current y, t values, differential variables, constants, a single differential component, a step size, and a number of steps
__global__ void rungaKutta43(double* inputs, double* constants, int steps, double stepSize, double** outputs) {
    //inputs, y0+h*k1/2, y0+h*k2/2, constants
    /*__shared__ double yValues[8 * 4];*/

    /*
    auto functions =
    {
            [] __device__(double* input, double* constant) {
            return -beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested);
        },
            []__device__(double* input, double* constant) {
            return -beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested) - alpha * exposed;
        },
            []__device__(double* input, double* constant) {
            return alpha * exposed - delta * presymptomatic;
        },
            []__device__(double* input, double* constant) {
            return delta * presymptomatic - (gamma + testingRate) * infectiousUntested;
        },
            []__device__(double* input, double* constant) {
            return testingRate * infectiousUntested - gamma * infectiousTested;
        },
            []__device__(double* input, double* constant) {
            InfectionFatalityRate
                return gamma * (1 - IFR) * infectiousUntested;
        },
            []__device__(double* input, double* constant) {
            InfectionFatalityRate
                return gamma * (1 - IFR) * infectiousTested;
        },
            []__device__(double* input, double* constant) {
            return 1 - susceptible - exposed - presymptomatic - infectiousTested - infectiousUntested - recoveredTested - recoveredUntested;
        }
    };
    yValues[threadIdx.x] = inputs[threadIdx.x];
    yValues[threadIdx.x + 8 * 3] = constants[threadIdx.x];
    __syncthreads();

    for (int i = 0; i < steps; ++i) {
        double k1 = stepSize * functions[threadIdx.x](&yValues[0], &yValues[8 * 3]) / 2;
        yValues[threadIdx.x + 8] = yValues[threadIdx.x] + k1;
        __syncthreads();
        double k2 = stepSize * functions[threadIdx.x](&yValues[8], &yValues[8 * 3]) / 2;
        yValues[threadIdx.x + 16] = yValues[threadIdx.x] + k2;
        __syncthreads();
        double k3 = stepSize * functions[threadIdx.x](&yValues[16], &yValues[8 * 3]);
        yValues[threadIdx.x + 24] = yValues[threadIdx.x] + k3;
        __syncthreads();
        double k4 = stepSize * functions[threadIdx.x](&yValues[0], &yValues[8 * 3]);
        yValues[threadIdx.x] += stepSize * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        outputs[i][threadIdx.x] = yValues[threadIdx.x];
        __syncthreads();
    }*/
}
