# Benchmark Report [[Results .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b2/all_results.csv) 

## Configuration

Length of signals: 1024

Repetitions: 2000

SNRin values: 
-5, 
0, 
5, 
10, 


### Methods  

* monte_carlo_test 

* global_mad_test 

* global_rank_env_test 

### Signals  

* LinearChirp 

## Mean results tables: 

The results shown here are the average and 95\% Clopper-Pearson CI of                             the estimated detection power with Bonferroni correction.                             Best performances are **bolded**. 
### Signal: LinearChirp[[View Plot]](https://jmiramont.github.io/signal-detection-benchmark/results/b2/plot_LinearChirp.html)    [[Get .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b2/results_LinearChirp.csv)
|    | Method + Param                                                                                                    | SNRin=-5dB (average)   | SNRin=-5dB (CI)   | SNRin=0dB (average)   | SNRin=0dB (CI)   | SNRin=5dB (average)   | SNRin=5dB (CI)   | SNRin=10dB (average)   | SNRin=10dB (CI)   |
|---:|:------------------------------------------------------------------------------------------------------------------|:-----------------------|:------------------|:----------------------|:-----------------|:----------------------|:-----------------|:-----------------------|:------------------|
|  0 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                     | 0.10                   | ['0.08', '0.11']  | 0.34                  | ['0.31', '0.37'] | 0.90                  | ['0.88', '0.92'] | **1.00**               | ['1.00', '1.00']  |
|  1 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                  | 0.12                   | ['0.10', '0.14']  | 0.65                  | ['0.62', '0.67'] | **1.00**              | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  2 | global_mad_test{'statistic': 'Frs', 'MC_reps': 199}                                                               | 0.09                   | ['0.07', '0.11']  | 0.31                  | ['0.28', '0.34'] | 0.86                  | ['0.84', '0.88'] | 1.00                   | ['1.00', '1.00']  |
|  3 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 199}                                                            | 0.11                   | ['0.09', '0.13']  | 0.65                  | ['0.62', '0.68'] | 1.00                  | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  4 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs'}                                                           | 0.10                   | ['0.08', '0.12']  | 0.57                  | ['0.54', '0.60'] | 1.00                  | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  5 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05}                               | **0.15**               | ['0.12', '0.17']  | 0.73                  | ['0.71', '0.76'] | 1.00                  | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  6 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'transform': 'asin(sqrt(.))'}                             | 0.11                   | ['0.09', '0.13']  | 0.58                  | ['0.55', '0.61'] | 1.00                  | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  7 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05, 'transform': 'asin(sqrt(.))'} | 0.14                   | ['0.12', '0.17']  | **0.74**              | ['0.71', '0.76'] | 1.00                  | ['1.00', '1.00'] | 1.00                   | ['1.00', '1.00']  |
