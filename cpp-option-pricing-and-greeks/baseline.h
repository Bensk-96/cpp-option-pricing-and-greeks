#pragma once

#include <algorithm>
#include <vector>
#include <cmath>


double gaussian_box_muller() {
    double x = 0.0;
    double y = 0.0;
    double euclid_sq = 0.0;

    do {
        x = 2.0 * rand() / static_cast<double>(RAND_MAX) - 1;
        y = 2.0 * rand() / static_cast<double>(RAND_MAX) - 1;
        euclid_sq = x * x + y * y;
    } while (euclid_sq >= 1.0);

    return x * sqrt(-2 * log(euclid_sq) / euclid_sq);
}

std::pair<double, double> option_price_baseline(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T)
{
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double S_cur = 0.0;
    double payoff_sum_call = 0.0;
    double payoff_sum_put = 0.0;

    for (int i = 0; i < num_sims; i++) {
        double gauss_bm = gaussian_box_muller();
        S_cur = S_adjust * exp(sqrt(v * v * T) * gauss_bm);
        payoff_sum_call += std::max(S_cur - K, 0.0);
        payoff_sum_put += std::max(K - S_cur, 0.0);
    }
    double call_price = (payoff_sum_call / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price = (payoff_sum_put / static_cast<double>(num_sims)) * exp(-r * T);

    std::pair<double, double> option_price = std::make_pair(call_price, put_price);

    return option_price;
}

std::pair<double, double> delta_baseline(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, const double& delta_S)
{
    double Splus = S + delta_S;
    double S_adjust = S * exp(T * (r - 0.5 * v * v));
    double Sp_adjust = Splus * exp(T * (r - 0.5 * v * v));

    double call_payoff_sum_p = 0.0;
    double call_payoff_sum = 0.0;
    double put_payoff_sum_p = 0.0;
    double put_payoff_sum = 0.0;

    for (int i = 0; i < num_sims; i++) {
        double gauss_bm = gaussian_box_muller(); // Generate the same random number for both
        double expgauss = exp(sqrt(v * v * T) * gauss_bm);

        double Sp_cur = Sp_adjust * expgauss;
        double S_cur = S_adjust * expgauss;

        call_payoff_sum_p += std::max(Sp_cur - K, 0.0);
        call_payoff_sum += std::max(S_cur - K, 0.0);
        put_payoff_sum_p += std::max(K - Sp_cur, 0.0);
        put_payoff_sum += std::max(K - S_cur, 0.0);
    }

    double call_price_sp = (call_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_S = (call_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_sp = (put_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_S = (put_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);

    double call_delta = (call_price_sp - call_price_S) / delta_S;
    double put_delta = (put_price_sp - put_price_S) / delta_S;

    std::pair<double, double> delta = std::make_pair(call_delta, put_delta);

    return delta;
}

std::pair<double, double> gamma_baseline(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T, const double& delta_S)
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

    for (int i = 0; i < num_sims; i++) {
        double gauss_bm = gaussian_box_muller(); // Use the same random number for all three paths
        double expgauss = exp(sqrt(v * v * T) * gauss_bm);

        double Sp_cur = Sp_adjust * expgauss;
        double S_cur = S_adjust * expgauss;
        double Sm_cur = Sm_adjust * expgauss;

        call_payoff_sum_p += std::max(Sp_cur - K, 0.0);
        call_payoff_sum += std::max(S_cur - K, 0.0);
        call_payoff_sum_m += std::max(Sm_cur - K, 0.0);

        put_payoff_sum_p += std::max(K - Sp_cur, 0.0);
        put_payoff_sum += std::max(K - S_cur, 0.0);
        put_payoff_sum_m += std::max(K - Sm_cur, 0.0);
    }

    double call_price_sp = (call_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_S = (call_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double call_price_sm = (call_payoff_sum_m / static_cast<double>(num_sims)) * exp(-r * T);

    double put_price_sp = (put_payoff_sum_p / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_S = (put_payoff_sum / static_cast<double>(num_sims)) * exp(-r * T);
    double put_price_sm = (put_payoff_sum_m / static_cast<double>(num_sims)) * exp(-r * T);

    double call_gamma = (call_price_sp - 2 * call_price_S + call_price_sm) / (delta_S * delta_S);
    double put_gamma = (put_price_sp - 2 * put_price_S + put_price_sm) / (delta_S * delta_S);

    std::pair<double, double> gamma = std::make_pair(call_gamma, put_gamma);

    return gamma;
}

