""" This script runs a benchmark with the parameters specified in the config.yaml file.

First, all methods are discovered in the "src/methods" folder.
Then a benchmark is initialized and run.
Results are saved in a serialized file with pickle (to be modified in later versions.)

Author: Juan Manuel Miramont
"""
if __name__ == "__main__":
    import importlib
    from src.methods import *
    from mcsm_benchs.benchmark_utils import MethodTemplate as MethodTemplate
    import time
    import inspect
    
    # Collects the methods in the folder/ module "methods" and make a global list
    print('Collecting methods to benchmark...')
    modules = dir()
    modules = [mod_name for mod_name in modules if mod_name.startswith('method_')]
    global list_of_methods # Global variable, use with caution.

    list_of_methods = list()    
    for mod_name in modules:
        mod = importlib.import_module('src.methods.' + mod_name)
        classes_in_mod = inspect.getmembers(mod, inspect.isclass)
        for a_class in classes_in_mod:
            method_class = getattr(mod, a_class[0])
            class_parent = method_class.__bases__[0]
            if class_parent == MethodTemplate:
                method_name = method_class().id
                print(method_name)
                list_of_methods.append(method_class())

    print('Done!')

    from mcsm_benchs.Benchmark import Benchmark
    import numpy as np
    from mcsm_benchs.ResultsInterpreter import ResultsInterpreter
    import yaml
    import pickle
    import os

    print('Configuring benchmark...')

    # Load parameters from configuration file.
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # config['task'] = 'detection'

    dictionary_of_methods = dict()
    dictionary_of_parameters = dict()

    # Select only methods for denoising.
    for method_instance in list_of_methods:
        if method_instance.task == config['task']:
            method_id = method_instance.id
            dictionary_of_methods[method_id] = method_instance.method
            dictionary_of_parameters[method_id] = method_instance.get_parameters()

    # Other parameters of the benchmark:    
    config['methods'] = dictionary_of_methods
    config['parameters'] = dictionary_of_parameters


    #-----------------------------------------------------------------------------------# If you want to use your own noise generating function
    #-----------------------------------------------------------------------------------
    # config['complex_noise'] = noise_fun(N,) # Import this from src/utilities

    #-----------------------------------------------------------------------------------
    # If you want to use your own performance metrics:
    #-----------------------------------------------------------------------------------
    # config['obj_fun'] = {'perf_metric_1': perf_metric_1(x, x_estimate, **kwargs), 
    #                      'perf_metric_2': perf_metric_2(x, x_estimate, **kwargs),
    #                     } 

    #-----------------------------------------------------------------------------------
    # If you want to use your own signals
    #-----------------------------------------------------------------------------------
    # config['signal_ids'] = {'Signal 1': signal_1, # signal_1 is a numpy array
    #                         'Signal 2': signal_2, # signal_2 is a numpy array
    #                         } 

    #-----------------------------------------------------------------------------------
    # Running the benchmark (don't touch this please!).
    #-----------------------------------------------------------------------------------
    if 'add_new_methods' in config.keys():
        if config['add_new_methods']:
            
            filename = os.path.join('results',config['task']+'_benchmark_results')
            with open(filename + '.pkl', 'rb') as f:
                benchmark = pickle.load(f)
            benchmark.add_new_method(config['methods'],config['parameters']) 
        else:
            config.pop('add_new_methods') 
            benchmark = Benchmark(**config)    
    else:
        benchmark = Benchmark(**config)   

    print('Done!')
    
    start = time.time()
    my_results = benchmark.run_test() # Run the test. my_results is a nested dictionary with the results for each of the variables of the simulation.
    end = time.time()
    print("The time of execution:", end-start)
    
    # Save the benchmark to a file. Notice that only the methods_ids are saved.
    filename = os.path.join('results',config['task']+'_benchmark_results')
    benchmark.save_to_file(filename = filename)
    
    df = benchmark.get_results_as_df() # This formats the results on a DataFrame
    print(df)
  


  
