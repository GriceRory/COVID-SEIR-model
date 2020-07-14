#define PICU 0.0125

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

__global__ void rungeKutta4(double* inputs, double* constants, int steps, double stepSize, double** simulation);
double** simulate(double* inputs, double* constants, int steps, double stepSize);

__global__ void rungeKutta4(double* inputs, double* constants, int steps, double stepSize, double** simulation) {
    //inputs for each layer = &yValues[0]
    //y0+h*k1/2 = &yValues[8]
    //y0+h*k2/2 = &yValues[16]
    //y0+h*k3 = &yValues[24]
    //constants = &yValues[32]
    __shared__ double yValues[8 * 5];
    yValues[threadIdx.x] = inputs[threadIdx.x];
    double* constant = &yValues[8 * 4];
    constant[threadIdx.x] = constants[threadIdx.x];
    simulation[0][threadIdx.x] = inputs[threadIdx.x];

    double k[4];
    double* input;

    //this is an intellisense bug, the compiler handles it just fine.
    __syncthreads();

    for (int step = 1; step < steps; ++step) {
        for (int kValue = 0; kValue < 3; ++kValue) {
            input = &yValues[8 * kValue];
            double IFR = 0.01;
            if (population * (infectiousTested + infectiousUntested) * PICU > ICUbeds) {
                IFR = 0.02 - 0.01 * ICUbeds / (population * (infectiousTested + infectiousUntested) * PICU);
            }

            if (threadIdx.x == 0) {
                //  dSuseptible/dt
                k[kValue] = stepSize * (-beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested));
            }
            if (threadIdx.x == 1) {
                //  dExposed/dt
                k[kValue] = stepSize * (beta * susceptible * (epsilon * presymptomatic + infectiousTested + infectiousUntested) - alpha * exposed);
            }
            if (threadIdx.x == 2) {
                //  dPresymptomatic/dt
                k[kValue] = stepSize * (alpha * exposed - delta * presymptomatic);
            }
            if (threadIdx.x == 3) {
                //  dInfectedUntested/dt
                k[kValue] = stepSize * (delta * presymptomatic - (gamma + testingRate) * infectiousUntested);
            }
            if (threadIdx.x == 4) {
                //  dInfectedTested/dt
                k[kValue] = stepSize * (testingRate * infectiousUntested - gamma * infectiousTested);
            }
            if (threadIdx.x == 5) {
                //  dRecoveredUntested/dt
                k[kValue] = stepSize * (gamma * infectiousUntested * (1 - IFR));
            }
            if (threadIdx.x == 6) {
                //  dRecoveredTested/dt
                k[kValue] = stepSize * (gamma * infectiousTested * (1 - IFR));
            }

            if (kValue < 2) { k[kValue] /= 2; }
            yValues[8 * (1 + kValue) + threadIdx.x] = yValues[threadIdx.x] + k[kValue];
            __syncthreads();
        }

        simulation[step][threadIdx.x] = simulation[step - 1][threadIdx.x] + (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6;
        if (threadIdx.x == 7) {
            //  Dead
            simulation[step][threadIdx.x] = (1 - susceptible - exposed - presymptomatic - infectiousTested - infectiousUntested - recoveredTested - recoveredUntested);
        }
        yValues[threadIdx.x] = simulation[step][threadIdx.x];
        __syncthreads();
    }
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

    rungeKutta4 <<<1, 8>>> (inputs, constants, steps, stepSize, d_simulation);

    for (int i = 0; i < steps; ++i) { cudaMemcpy(h_simulation[i], d_simulation[i], 8 * sizeof(double), cudaMemcpyDeviceToHost); }
    return h_simulation;
}

