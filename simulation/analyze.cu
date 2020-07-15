
#define BLOCK_SIZE 32*2


typedef struct{
	int degree;
	double *coefficiants;
}polynomial;

typedef struct {
	double base;
	double baseConstant;
	double exponentConstant;
	double exponentDisplacement;
}exponential;

typedef struct {
	polynomial* polynomials;
	exponential* decays;
	int length;
}trendFunction;

__device__ void reduce(double *reduction);
double error(double* expected, double* actual);
polynomial fitPolynomail(double* data, int degree);
polynomial differentiate(polynomial p);
trendFunction calculateTrend(double* data, int polynomialDegree, int polynomialRange, int dataLength);
double trendAtTime(trendFunction trend, double time);
double polynomialAtTime(polynomial p, double time);
double exponentialAtTime(exponential e, double time);

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

double polynomialAtTime(polynomial p, double time) {
	double value = 0;
	double t = 1;
	for (int i = 0; i < p.degree; ++i) {
		value += p.coefficiants[p.degree-i] * t;
		t *= time;
	}
	return value;
}
double exponentialAtTime(exponential e, double time) {
	double value = 0;

	return e.baseConstant * pow(e.base, e.exponentConstant * (time - e.exponentDisplacement));
}

trendFunction calculateTrend(double* data, int polynomialDegree, int polynomialRange, int dataLength) {
	trendFunction trend;
	trend.length = dataLength - polynomialRange;
	cudaMallocHost(&trend.decays, trend.length * sizeof(exponential));
	cudaMallocHost(&trend.polynomials, trend.length * sizeof(polynomial));
	for (int poly = 0; poly < trend.length; ++poly) {
		polynomial p = fitPolynomail(&data[poly], polynomialDegree);
		trend.polynomials[poly] = p;
		exponential e;
		e.base = 2.81;
		e.exponentDisplacement = -poly;
		e.exponentConstant = -1.0/polynomialRange;
		e.baseConstant = 1.0 / polynomialRange;
		trend.decays[poly] = e;
	}
	return trend;
}



double trendAtTime(trendFunction trend, double time) {
	double value = 0;
	for (int i = 0; i < trend.length; ++i) {
		value += polynomialAtTime(trend.polynomials[i], i) * exponentialAtTime(trend.decays[i], i);
	}
	return value;
}


