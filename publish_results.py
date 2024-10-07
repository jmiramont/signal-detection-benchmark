""" This script generates interactive figures and files from the results of all the benchmarks saved in the folder "results", and prepare them for publication in a GitHub-hosted website.

To be run by GitHub Actions (username and repo name are given as input arguments).

Author: Juan Manuel Miramont

"""
if __name__ == "__main__":
    # import importlib
    from src.methods import *
    from mcsm_benchs.benchmark_utils import MethodTemplate as MethodTemplate
    # import inspect

    from mcsm_benchs.Benchmark import Benchmark
    # import numpy as np
    from mcsm_benchs.ResultsInterpreter import ResultsInterpreter
    # import yaml
    # import pickle
    import os
    
    print('Getting benchmarks paths...')
    paths = []
    for file in os.listdir('results'):
        print(file, file[-4::])
        if file[-4::]== '.pkl':
            print(file[:-4])
            paths.append(file[:-4])
    
    sub_folders = ['b{}'.format(i+1) for i,p in enumerate(paths)]

    # Write a .md table to summarize the results:
    output_path = os.path.join('results','index.md')
    
    # Generate table header:
    lines = ['# Results',   '| Benchmark | Description | Link | ',
                            '| --------- | ----------- | ---- | \n']
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    # For each benchmark in src/results
    for file, sub_folder in zip(paths,sub_folders):

        # Load Benchmark
        filename=os.path.join('results',file)
        try:
            benchmark = Benchmark.load_benchmark(filename=filename)
            interpreter = ResultsInterpreter(benchmark)

        except BaseException as err:
            print(f"Unexpected error {err=}, {type(err)=} loading {filename}.")

        # Get environment information from GitHub Actions (if possible):
        try:
            username = os.environ['OWNER']
            reponame = os.environ['NAME']
            print("Owner:", username, " Repo:", reponame)

        except BaseException as err:
            print(f"Unexpected error {err=}, {type(err)=} loading GitHub env.")
            username='USERNAME'
            reponame='MY-REPO'
        
        link = 'https://{}.github.io/{}/results/{}'.format(username,reponame,sub_folder)
        print(link)


        # Report shown in the repo 
        interpreter.save_report(filename='results_{}.md'.format(sub_folder),
                                path=os.path.join('results'), 
                                link=link)

        # Interactive figures shown in the repo
        interpreter.get_html_figures(df=interpreter.get_benchmark_as_data_frame(),
                                    #   varfun=cp_ci, 
                                    path=os.path.join('results',sub_folder), 
                                    bars=True, 
                                    ylabel='Perf. Fun.',
                                    )

        # .csv files for sharing results
        interpreter.get_csv_files(path=os.path.join('results',sub_folder),)

        # Append row under header
        link2 = 'https://{}.github.io/{}/'.format(username,reponame)

        if benchmark.name is None:
            table_string = '| Benchmark: {}'.format(sub_folder)
        else:
            table_string = '| {}'.format(benchmark.name) 

        if benchmark.description is None:
            table_string = table_string + '| '
        else:
            table_string = table_string + '| '+ benchmark.description

        table_string = table_string + ' | [Link]('+link2+'results_{}.html) | \n'.format(sub_folder)

        with open(output_path, 'a') as f:
            f.write(table_string)

        #TODO Build table to summarize the results.

