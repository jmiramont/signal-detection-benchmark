# Benchmark Report [[Results .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b1/all_results.csv) 

## Configuration

Length of signals: 256

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
### Signal: LinearChirp[[View Plot]](https://jmiramont.github.io/signal-detection-benchmark/results/b1/plot_LinearChirp.html)    [[Get .csv]](https://jmiramont.github.io/signal-detection-benchmark/results/b1/results_LinearChirp.csv)
|    | Method + Param                                                                                                    | SNRin=-5dB (average)   | SNRin=-5dB (CI)   | SNRin=0dB (average)   | SNRin=0dB (CI)   | SNRin=5dB (average)   | SNRin=5dB (CI)   | SNRin=10dB (average)   | SNRin=10dB (CI)   |
|---:|:------------------------------------------------------------------------------------------------------------------|:-----------------------|:------------------|:----------------------|:-----------------|:----------------------|:-----------------|:-----------------------|:------------------|
|  0 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                    | 0.05                   | ['0.03', '0.06']  | 0.07                  | ['0.06', '0.09'] | 0.14                  | ['0.12', '0.17'] | 0.29                   | ['0.26', '0.32']  |
|  1 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 0.5, 'MC_reps': 2499}                                 | 0.04                   | ['0.03', '0.06']  | 0.07                  | ['0.06', '0.09'] | 0.14                  | ['0.12', '0.16'] | 0.28                   | ['0.25', '0.31']  |
|  2 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                    | 0.06                   | ['0.04', '0.07']  | 0.17                  | ['0.14', '0.19'] | 0.65                  | ['0.61', '0.68'] | 1.00                   | ['1.00', '1.00']  |
|  3 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 1.0, 'MC_reps': 2499}                                 | 0.07                   | ['0.06', '0.09']  | 0.27                  | ['0.24', '0.30'] | 0.94                  | ['0.92', '0.95'] | **1.00**               | ['1.00', '1.00']  |
|  4 | monte_carlo_test{'statistic': 'Frs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                    | 0.06                   | ['0.05', '0.08']  | 0.17                  | ['0.15', '0.20'] | 0.65                  | ['0.62', '0.68'] | 1.00                   | ['1.00', '1.00']  |
|  5 | monte_carlo_test{'statistic': 'Frs_vs', 'pnorm': 2, 'rmax': 2.0, 'MC_reps': 2499}                                 | 0.07                   | ['0.06', '0.09']  | 0.27                  | ['0.25', '0.30'] | 0.94                  | ['0.92', '0.95'] | 1.00                   | ['1.00', '1.00']  |
|  6 | global_mad_test{'statistic': 'Frs', 'MC_reps': 2499}                                                              | 0.06                   | ['0.04', '0.07']  | 0.16                  | ['0.14', '0.18'] | 0.65                  | ['0.61', '0.68'] | 1.00                   | ['1.00', '1.00']  |
|  7 | global_mad_test{'statistic': 'Frs_vs', 'MC_reps': 2499}                                                           | 0.07                   | ['0.05', '0.08']  | 0.27                  | ['0.24', '0.30'] | 0.96                  | ['0.95', '0.97'] | 1.00                   | ['1.00', '1.00']  |
|  8 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs'}                                                           | 0.07                   | ['0.05', '0.08']  | 0.20                  | ['0.17', '0.22'] | 0.91                  | ['0.89', '0.93'] | 1.00                   | ['1.00', '1.00']  |
|  9 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05}                               | **0.09**               | ['0.07', '0.11']  | **0.34**              | ['0.31', '0.37'] | **0.97**              | ['0.96', '0.98'] | 1.00                   | ['1.00', '1.00']  |
| 10 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'transform': 'asin(sqrt(.))'}                             | 0.06                   | ['0.05', '0.08']  | 0.20                  | ['0.18', '0.23'] | 0.91                  | ['0.88', '0.92'] | 1.00                   | ['1.00', '1.00']  |
| 11 | global_rank_env_test{'fun': 'Fest', 'correction': 'rs', 'rmin': 0.65, 'rmax': 1.05, 'transform': 'asin(sqrt(.))'} | 0.09                   | ['0.07', '0.11']  | 0.34                  | ['0.31', '0.37'] | 0.97                  | ['0.96', '0.98'] | 1.00                   | ['1.00', '1.00']  |
