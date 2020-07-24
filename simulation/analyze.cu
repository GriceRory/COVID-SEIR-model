#include <cuda_runtime.h>
#include "linear_algebra.cu"


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
polynomial fitPolynomail(double* data, double* inputs, int degree);
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

polynomial fitPolynomail(double* data, double* inputs, int degree, int dataLength) {
	polynomial p;
	matrix *h_X, *h_Xt, *h_inverse, *h_temp;
	matrix *d_X, *d_Xt, *d_inverse, *d_temp;
	vector *h_inputs, *d_inputs, *h_data, *d_data;

	h_X = cuda_build_matrix(dataLength, degree);
	h_Xt = cuda_build_matrix(degree, dataLength);
	h_inverse = cuda_build_matrix(degree, degree);
	h_temp = cuda_build_matrix(degree, degree);
	h_inputs = build_vector(degree);
	h_data = build_vector(degree);

	d_X = cuda_build_matrix(dataLength, degree);
	d_Xt = cuda_build_matrix(degree, dataLength);
	d_inverse = cuda_build_matrix(degree, degree);
	d_temp = cuda_build_matrix(degree, degree);
	d_inputs = cuda_build_vector(degree);
	d_data = cuda_build_vector(degree);
		
	copy_host_to_device(h_X, d_X);
	copy_host_to_device(h_Xt, d_Xt);
	copy_host_to_device(h_inverse, d_inverse);
	copy_host_to_device(h_temp, d_temp);


	matrix_multiply(d_Xt, d_X, d_temp);
	invert<<<1, d_temp->height>>>(d_temp, d_inverse);
	//constants = (X^T X)^-1 * X^T * Y

	cuda_free_matrix(d_X);
	cuda_free_matrix(d_Xt);
	cuda_free_matrix(d_inverse);
	
	
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

	return e.baseConstant * pow(e.base, e.exponentConstant * (time - e.exponentDisplacement) * (time - e.exponentDisplacement));
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
		e.baseConstant = 1.2 / polynomialRange;
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


