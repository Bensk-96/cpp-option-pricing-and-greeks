#pragma once

#include <mkl.h>
#include <omp.h>
#include <immintrin.h> 


const int BATCH_SIZE = 1024;  // Define the size of each batch for random number generation

std::pair<double, double> option_price_mkl(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T)
{
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double payoff_sum_call = 0.0;
    double payoff_sum_put = 0.0;
    double sqrt_v_T = sqrt(v * v * T);

    // Allocate aligned memory for random numbers (aligned to 64 bytes for AVX/AVX2)
    double* rand_nums = (double*)_mm_malloc(BATCH_SIZE * sizeof(double), 64);

    if (rand_nums == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return std::make_pair(0.0, 0.0);  // Handle allocation failure
    }

    // Initialize MKL random stream
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 12345);  // Mersenne Twister RNG with seed 12345

    // Use OpenMP to parallelize the batch processing
#pragma omp parallel reduction(+:payoff_sum_call, payoff_sum_put)
    {
        // Each thread processes its own batch
        int thread_id = omp_get_thread_num();
        VSLStreamStatePtr local_stream;
        vslNewStream(&local_stream, VSL_BRNG_MT2203 + thread_id, 12345);

#pragma omp for schedule(static)
        for (int i = 0; i < num_sims; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_sims - i);

            // Generate a batch of Gaussian random numbers
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, local_stream, current_batch_size, rand_nums, 0.0, 1.0);

            // Process the current batch
            for (int j = 0; j < current_batch_size; j++) {
                double S_cur = S_adjust * exp(sqrt_v_T * rand_nums[j]);
                payoff_sum_call += std::max(S_cur - K, 0.0);
                payoff_sum_put += std::max(K - S_cur, 0.0);
            }
        }

        // Delete local MKL random stream after use
        vslDeleteStream(&local_stream);
    }

    // Free MKL random stream and aligned memory
    vslDeleteStream(&stream);
    _mm_free(rand_nums);  // Free the aligned memory

    // Calculate option prices
    double call_price = (payoff_sum_call / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price = (payoff_sum_put / static_cast<double>(num_sims)) * exp(-r * T);

    // Return the results as a pair
    return std::make_pair(call_price, put_price);
}

std::pair<double, double> delta_mkl(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, const double& delta_S)
{
    double Splus = S + delta_S;
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double Sp_adjust = Splus * exp(T * (r - 0.5 * v * v));

    double call_payoff_sum_p = 0.0;
    double call_payoff_sum = 0.0;
    double put_payoff_sum_p = 0.0;
    double put_payoff_sum = 0.0;
    double sqrt_v_T = sqrt(v * v * T);

    // Allocate aligned memory for random numbers (aligned to 64 bytes for AVX/AVX2)
    double* rand_nums = (double*)_mm_malloc(BATCH_SIZE * sizeof(double), 64);

    if (rand_nums == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return std::make_pair(0.0, 0.0);  // Handle allocation failure
    }

    // Initialize MKL random stream
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 12345);  // Mersenne Twister RNG with seed 12345

    // Use OpenMP to parallelize the batch processing
#pragma omp parallel reduction(+:call_payoff_sum_p, call_payoff_sum,put_payoff_sum_p,put_payoff_sum)
    {
        // Each thread processes its own batch
        int thread_id = omp_get_thread_num();
        VSLStreamStatePtr local_stream;
        vslNewStream(&local_stream, VSL_BRNG_MT2203 + thread_id, 12345);

#pragma omp for schedule(static)
        for (int i = 0; i < num_sims; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_sims - i);

            // Generate a batch of Gaussian random numbers
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, local_stream, current_batch_size, rand_nums, 0.0, 1.0);

            // Process the current batch
            for (int j = 0; j < current_batch_size; j++) {
                double S_cur = S_adjust * exp(sqrt_v_T * rand_nums[j]);
                double Sp_cur = Sp_adjust * exp(sqrt_v_T * rand_nums[j]);
                call_payoff_sum_p += std::max(Sp_cur - K, 0.0);
                call_payoff_sum += std::max(S_cur - K, 0.0);
                put_payoff_sum_p += std::max(K - Sp_cur, 0.0);
                put_payoff_sum += std::max(K - S_cur, 0.0);
            }
        }

        // Delete local MKL random stream after use
        vslDeleteStream(&local_stream);
    }

    // Free MKL random stream and aligned memory
    vslDeleteStream(&stream);
    _mm_free(rand_nums);  // Free the aligned memory

    // Calculate option prices
    double call_price_sp = (call_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_S = (call_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_sp = (put_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_S = (put_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);

    double call_delta = (call_price_sp - call_price_S) / delta_S;
    double put_delta = (put_price_sp - put_price_S) / delta_S;

    // Return the results as a pair
    return std::make_pair(call_delta, put_delta);
}

std::pair<double, double> gamma_mkl(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, const double& delta_S)
{

    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double Sp_adjust = (S + delta_S) * exp(T * (r - 0.5 * v * v));
    double Sm_adjust = (S - delta_S) * exp(T * (r - 0.5 * v * v));

    double call_payoff_sum_p = 0.0;
    double call_payoff_sum = 0.0;
    double call_payoff_sum_m = 0.0;

    double put_payoff_sum_p = 0.0;
    double put_payoff_sum = 0.0;
    double put_payoff_sum_m = 0.0;

    double sqrt_v_T = sqrt(v * v * T);

    // Allocate aligned memory for random numbers (aligned to 64 bytes for AVX/AVX2)
    double* rand_nums = (double*)_mm_malloc(BATCH_SIZE * sizeof(double), 64);

    if (rand_nums == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return std::make_pair(0.0, 0.0);  // Handle allocation failure
    }

    // Initialize MKL random stream
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MT19937, 12345);  // Mersenne Twister RNG with seed 12345

    // Use OpenMP to parallelize the batch processing
#pragma omp parallel reduction(+:call_payoff_sum_p, call_payoff_sum, call_payoff_sum_m, put_payoff_sum_p, put_payoff_sum, put_payoff_sum_m)
    {
        // Each thread processes its own batch
        int thread_id = omp_get_thread_num();
        VSLStreamStatePtr local_stream;
        vslNewStream(&local_stream, VSL_BRNG_MT2203 + thread_id, 12345);

#pragma omp for schedule(static)
        for (int i = 0; i < num_sims; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_sims - i);

            // Generate a batch of Gaussian random numbers
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, local_stream, current_batch_size, rand_nums, 0.0, 1.0);


            // Process the current batch
            for (int j = 0; j < current_batch_size; j++) {
                double S_cur = S_adjust * exp(sqrt_v_T * rand_nums[j]);
                double Sp_cur = Sp_adjust * exp(sqrt_v_T * rand_nums[j]);
                double Sm_cur = Sm_adjust * exp(sqrt_v_T * rand_nums[j]);

                call_payoff_sum_p += std::max(Sp_cur - K, 0.0);
                call_payoff_sum += std::max(S_cur - K, 0.0);
                call_payoff_sum_m += std::max(Sm_cur - K, 0.0);

                put_payoff_sum_p += std::max(K - Sp_cur, 0.0);
                put_payoff_sum += std::max(K - S_cur, 0.0);
                put_payoff_sum_m += std::max(K - Sm_cur, 0.0);
            }
        }

        // Delete local MKL random stream after use
        vslDeleteStream(&local_stream);
    }

    // Free MKL random stream and aligned memory
    vslDeleteStream(&stream);
    _mm_free(rand_nums);  // Free the aligned memory

    // Calculate option prices
    double call_price_sp = (call_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_S = (call_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_sm = (call_payoff_sum_m / static_cast<double>(num_sims)) * exp(-r * T);

    double put_price_sp = (put_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_S = (put_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_sm = (put_payoff_sum_m / static_cast<double>(num_sims)) * exp(-r * T);

    double call_gamma = (call_price_sp - 2 * call_price_S + call_price_sm) / (delta_S * delta_S);
    double put_gamma = (put_price_sp - 2 * put_price_S + put_price_sm) / (delta_S * delta_S);

    // Return the results as a pair
    return std::make_pair(call_gamma, put_gamma);
}


