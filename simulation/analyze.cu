
#define BLOCK_SIZE 32*2

typedef struct{
	int degree;
	double *coefficiants;
}polynomial;

typedef struct {
	double base;
	double exponentConstant;
	double exponentDisplacement;
}exponential;

typedef struct {
	polynomial* polynomials;
	exponential* decays;
}trendFunction;

__device__ void reduce(double *reduction);
double error(double* expected, double* actual);
polynomial fitPolynomail(double* data, int degree);
polynomial differentiate(polynomial p);
trendFunction calculateTrend(double* data);
double trendAtTime(trendFunction trend, double time);

/*
__device__ void reduce(double *reduction) {
	
}

double error(double* expected, double* actual) {
	return 0;
}


__global__ void error(double* expected, double* actual, double* error, int length) {
	__shared__ double errors[BLOCK_SIZE];
	errors[threadIdx.x] = 0;

	for (int location = 0; location + threadIdx.x < (double)length / BLOCK_SIZE; ++location) {
		errors[threadIdx.x] += (actual[threadIdx.x] - expected[threadIdx.x]) * (actual[threadIdx.x] - expected[threadIdx.x]);
	}
	reduce(errors);
	erorr[0] = errors[0];
}*/

polynomial fitPolynomail(double* data, int degree) {
	polynomial p;
	return p;
}

polynomial differentiate(polynomial p) {
	polynomial pPrime;
	pPrime.degree = p.degree - 1;
	cudaMallocHost(&pPrime.coefficiants, sizeof(double)*pPrime.degree);
	for (int coefficient = 0; coefficient < pPrime.degree; ++coefficient) {
		pPrime.coefficiants[coefficient] = coefficient * p.coefficiants[coefficient];
	}
	return pPrime;
}

trendFunction calculateTrend(double* data) {
	trendFunction trend;
	return trend;
}

double trendAtTime(trendFunction trend, double time) {
	double value;
	return value;
}


