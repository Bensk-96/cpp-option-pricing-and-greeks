#include <iostream>
#include <chrono>

#include "baseline.h"
#include "openmp.h"
#include "intelmkl.h"



int main(int argc, char** argv) {
	std::cout << "Monte Carlo Option Pricing and Greeks Calculation" <<  std::endl;
	int available_threads = omp_get_max_threads();
	omp_set_num_threads(available_threads);
	std::cout << "Number of cores is " << available_threads << std::endl;
	printf("----------------------------------------------------------------- \n");

	int num_sims = 10000000;
	double S = 100.0;
	double delta_S = 0.001;
	double K = 100.0;
	double r = 0.05;
	double v = 0.2;
	double T = 1.0;

	std::cout << "number of paths: " << num_sims << std::endl;
	std::cout << "underlying:      " << S << std::endl;
	std::cout << "strike:          " << K << std::endl;
	std::cout << "risk-free rate:  " << r << std::endl;
	std::cout << "volatility:      " << v << std::endl;
	std::cout << "maturity:        " << T << std::endl;
	printf("----------------------------------------------------------------- \n");

	std::cout << "Running baseline monte carlo..." << std::endl;
	std::chrono::high_resolution_clock::time_point start_bl = std::chrono::high_resolution_clock::now();

	std::pair<double, double> price_bl = option_price_baseline(num_sims, S, K, r, v, T);
	std::pair<double, double> delta_bl = delta_baseline(num_sims, S, K, r, v, T, delta_S);
	std::pair<double, double> gamma_bl = gamma_baseline(num_sims, S, K, r, v, T, delta_S);

	std::chrono::high_resolution_clock::time_point end_bl = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span_bl = std::chrono::duration_cast<std::chrono::duration<double>>(end_bl - start_bl);
	std::cout << "runtime is " << time_span_bl.count() << " seconds. \n";
	printf("----------------------------------------------------------------- \n");

	std::cout << "Running OpenMP Monte Carlo..." << std::endl;
	std::chrono::high_resolution_clock::time_point start_omp = std::chrono::high_resolution_clock::now();

	std::pair<double, double> price_openmp = option_price_omp(num_sims, S, K, r, v, T);
	std::pair<double, double> delta_openmp = delta_omp(num_sims, S, K, r, v, T, delta_S);
	std::pair<double, double> gamma_openmp = gamma_omp(num_sims, S, K, r, v, T, delta_S);

	std::chrono::high_resolution_clock::time_point end_omp = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span_omp = std::chrono::duration_cast<std::chrono::duration<double>>(end_omp - start_omp);
	std::cout << "Runtime is " << time_span_omp.count() << " seconds. \n";
	std::cout << "Speedup is " <<  time_span_bl.count() / time_span_omp.count() << " times. \n";
	printf("----------------------------------------------------------------- \n");

	std::cout << "Running Intel MKL + OpenMP Monte Carlo..." << std::endl;
	std::chrono::high_resolution_clock::time_point start_mkl = std::chrono::high_resolution_clock::now();

	std::pair<double, double> price_intel_mkl = option_price_mkl(num_sims, S, K, r, v, T);
	std::pair<double, double> delta_intel_mkl = delta_mkl(num_sims, S, K, r, v, T, delta_S);
	std::pair<double, double> gamma_intel_mkl = gamma_mkl(num_sims, S, K, r, v, T, delta_S);

	std::chrono::high_resolution_clock::time_point end_mkl = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span_mkl = std::chrono::duration_cast<std::chrono::duration<double>>(end_mkl - start_mkl);
	std::cout << "Runtime is " << time_span_mkl.count() << " seconds. \n";
	std::cout << "Speedup is " <<  time_span_bl.count() / time_span_mkl.count()  << " times. \n";
	printf("----------------------------------------------------------------- \n");

	std::cout << "Call Price" << std::endl;
	std::cout << "Baseline: " << price_bl.first << ", OpenMP: " << price_openmp.first << ", Intel MKL + OpenMP: " << price_intel_mkl.first << std::endl;
	
	std::cout << "Put Price" << std::endl;
	std::cout << "Baseline: " << price_bl.second << ", OpenMP: " << price_openmp.second << ", Intel MKL + OpenMP: " << price_intel_mkl.second << std::endl;
	
	std::cout << "Call Delta" << std::endl;
	std::cout << "Baseline: " << delta_bl.first << ", OpenMP: " << delta_openmp.first << ", Intel MKL + OpenMP: " << delta_intel_mkl.first << std::endl;
	
	std::cout << "Put Delta" << std::endl;
	std::cout << "Baseline: " << delta_bl.second << ", OpenMP: " << delta_openmp.second << ", Intel MKL + OpenMP: " << delta_intel_mkl.second << std::endl;
	
	std::cout << "Call Gamma" << std::endl;
	std::cout << "Baseline: " << gamma_bl.first << ", OpenMP: " << gamma_openmp.first << ", Intel MKL + OpenMP: " << gamma_intel_mkl.first << std::endl;

	std::cout << "Put Gamma" << std::endl;
	std::cout << "Baseline: " << gamma_bl.second << ", OpenMP: " << gamma_openmp.second << ", Intel MKL + OpenMP: " << gamma_intel_mkl.second << std::endl;
	printf("----------------------------------------------------------------- \n");


	return 0;
}