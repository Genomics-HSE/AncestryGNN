import ancinf
import json
import os
import subprocess

def run_console_command(command):
    process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    lst_lines = []
    for line in process.stdout:
        print(line, end='')
        lst_lines.append(line)
    return lst_lines    
    process.wait()
    
def repeat_10(test_name):
    command1 = f"python3 -m ancinf simulate ./test/auto/group_comparison/ --infile {test_name}.params"
    repeat_n = 2
    
    def check_people_entry(arrakis_data, arrakis_init_data):
        pop_names = arrakis_init_data['datasets']['Arrakis']['pop_names']
        pop_sizes = arrakis_init_data['datasets']['Arrakis']['pop_sizes']
        split_count = arrakis_init_data['crossvalidation']['split_count']

        lst_pop = []

        for i in range(split_count):
            temp = []
            for j in range(len(pop_names)):
                temp.append([])
            lst_pop.append(temp)

        for i in range(split_count):
            for j in range(len(pop_names)):
                lst_pop[i][j] += arrakis_data['partitions'][i]['test'][pop_names[j]]
                lst_pop[i][j] += arrakis_data['partitions'][i]['train'][pop_names[j]]
                lst_pop[i][j] += arrakis_data['partitions'][i]['val'][pop_names[j]]            
        return(lst_pop)

    def run_n_times(n):
        lst_repeat = []
        for i in range(n):
            run_console_command(command1)
            parent_dir = os.path.dirname(os.path.abspath(__file__))

            with open(f"./test/auto/group_comparison/{test_name}_Arrakis_exp0.split", "r") as f:
                arrakis_data1 = json.load(f)

            with open(f"./test/auto/group_comparison/{test_name}.params", "r") as f1:
                arrakis_init_data1 = json.load(f1)

            lst_repeat.append(check_people_entry(arrakis_data1, arrakis_init_data1))
        return(lst_repeat)

    lst_repeat = run_n_times(repeat_n)
    count = 0
    for i in range(len(lst_repeat)):
        if lst_repeat[i] == lst_repeat[0]:
                   count += 1
    if count == repeat_n:
        assert True, "Массивы испытуемых совпали"
    else:
        assert False, "Массивы испытуемых не совпали"
        

def test_helook():
    # python3 -m ancinf preprocess ../datasets/ ./test5/ --infile simple.ancinf
    ancinf.preprocess_fn("./test/datasets/", "./test/manual/test5/", "simple.ancinf", None, 2023)
    # python3 -m ancinf crossval ./test5/ --infile simple.explist 
    # workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount
    ancinf.crossval_fn("./test/manual/test5/", "simple.explist", None, 2023, 1, None, None, None, None, 0, 1)
    assert True 

def test_comparison_parameters_10_100():
    command1 = f"python3 -m ancinf simulate ./test/auto/test_comparison_parameters_10_100/ --infile arrakis100-10.params"   
    command2 = f"python3 -m ancinf getparams ./test/auto/test_comparison_parameters_10_100/ --infile simple.ancinf"
    command3 = f"python3 -m ancinf getparams ./test/auto/test_comparison_parameters_10_100/ --infile simple --outfile arrakis100-10_new.params"
    command4 = f"diff ./test/auto/test_comparison_parameters_10_100/arrakis100-10.params ./test/auto/test_comparison_parameters_10_100/arrakis100-10_new.params"
    run_console_command(command1)
    run_console_command(command2)
    lst = run_console_command(command4)
    if lst == []:
        assert True, "Params files are equal"
    else:
        assert False, "Params files are not equal"
    
def test_group_comparison_10():
    test_name = 'arrakis10'
    repeat_10(test_name)
    
def test_group_comparison_100():
    test_name = 'arrakis100'
    repeat_10(test_name)
    
def test_group_comparison_10_100():
    test_name = 'arrakis10-100'
    repeat_10(test_name)
    
def test_group_comparison_10_100_100():
    test_name = 'arrakis10-100-100'
    repeat_10(test_name)
    
    