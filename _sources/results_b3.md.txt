# Benchmark Report [[Results .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b3/all_results.csv) 

## Configuration

Length of signals: 512

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
### Signal: LinearChirp[[View Plot]](https://jmiramont.github.io/signal-detection-benchmark/results/b3/plot_LinearChirp.html)    [[Get .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b3/results_LinearChirp.csv)
|    | Method + Param                                                                                                    | SNRin=-5dB (average)   | SNRin=-5dB (CI)   | SNRin=0dB (average)   | SNRin=0dB (CI)   | SNRin=5dB (average)   | SNRin=5dB (CI)   | SNRin=10dB (average)   | SNRin=10dB (CI)   |
|---:|:------------------------------------------------------------------------------------------------------------------|:-----------------------|:------------------|:----------------------|:-----------------|:----------------------|:-----------------|:-----------------------|:------------------|
|  0 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                     | 0.08                   | ['0.06', '0.10']  | 0.23                  | ['0.20', '0.26'] | 0.80                  | ['0.78', '0.83'] | **1.00**               | ['1.00', '1.00']  |
|  1 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 199}                                  | 0.09                   | ['0.08', '0.11']  | 0.42                  | ['0.39', '0.45'] | 0.99                  | ['0.99', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  2 | global_mad_test{'statistic': 'Frs', 'MC_reps': 199}                                                               | 0.07                   | ['0.06', '0.09']  | 0.23                  | ['0.20', '0.25'] | 0.78                  | ['0.75', '0.80'] | 1.00                   | ['1.00', '1.00']  |
|  3 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 199}                                                            | 0.08                   | ['0.07', '0.10']  | 0.42                  | ['0.39', '0.45'] | 1.00                  | ['0.99', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  4 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs'}                                                           | 0.08                   | ['0.06', '0.10']  | 0.31                  | ['0.29', '0.34'] | 0.99                  | ['0.98', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  5 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05}                               | 0.11                   | ['0.09', '0.13']  | **0.49**              | ['0.46', '0.52'] | **1.00**              | ['0.99', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  6 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'transform': 'asin(sqrt(.))'}                             | 0.07                   | ['0.06', '0.09']  | 0.31                  | ['0.28', '0.34'] | 0.99                  | ['0.99', '1.00'] | 1.00                   | ['1.00', '1.00']  |
|  7 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05, 'transform': 'asin(sqrt(.))'} | **0.11**               | ['0.10', '0.13']  | 0.49                  | ['0.46', '0.52'] | 1.00                  | ['0.99', '1.00'] | 1.00                   | ['1.00', '1.00']  |
