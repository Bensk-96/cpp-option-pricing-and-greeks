# Monte Carlo Simulation for Option Pricing and Greeks Calculation

The goal of this project is to price Vanilla European options and calculate the underlying greeks(Delta and Gamma) using **Monte Carlo simulatin** combined with **finite difference method**.
Monte Carlo simulation is idea for parallelizationb since it involves generating numerous indepdent paths. This project accelerates the computations using both **OpenMP** (for parallel processing) and **Intel MKl** (a highly optimized math library that utilizes SIMD instructions for accelerated performance).

## Project Structure 

- **baseline.h** Baseline calculation using sequential MOnte Carlo
- **openmp.h**   Parallelized Monte Carlo simulation using OpenMP.
- **intelmkl.h** Optimize Monte Carlo simulation using both Intel MKL and OpenMP.
- **main.cpp**   Runs the option pricing and Greeks calculation across the methods above and record the runtimes for comparison.

## System Specifications

CPU: Intel Core™ i7-12700H Processor with 14 cores and 20 threads 
Compiler: Intel C/C++ Compiler

## Result

For this project, we simulate 10,000,000 paths to calculate the option price. Calculating Delta and Gamma requires 2 and 3 times the paths, respectively, due to the finite difference method. Thus, a total of 60,000,000 paths are processed for a complete set of option pricing and Greek calculations.

![image](https://github.com/user-attachments/assets/42f46092-8101-4bef-a5d1-dcdb1ad02d54)

- The baseline computation takes 1.2 seconds, which is significantly fast, especcially compared to high-level programming languages like Python.
- Using OpenMP achieves a 3.5x speedup.
- Combining Intel MKL with OpenMP yields an impressive 41x speedup. The largest gain comes from the optimized random number generation methods in the MKL library, which are crucial for Monte Carlo simulation performance.


## Reference
- https://www.quantstart.com/articles/European-vanilla-option-pricing-with-C-via-Monte-Carlo-methods/
- https://www.intel.com/content/www/us/en/docs/onemkl/cookbook/2023-1/monte-carlo-simulating-european-options-pricing.html

