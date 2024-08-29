import click
import numpy as np
import time
from multiprocessing import Pool

from .utils import simulate as sim
import json
import os
import glob

from .utils.ibdloader import data_stats


@click.group()
def cli():
    pass


# STAGE1 GETPARAMS
def getparams_fn(datadir, workdir, infile, outfile):
    if outfile is None:
        # try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position > 0:
            outfile = infile[:position] + '.params'
        else:
            outfile = infile + '.params'
    sim.collectandsaveparams(datadir, workdir, infile, outfile)
    print("Finished!")


@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with simulation parameters, "
                                              "defaults to project file with '.params' extension")
def getparams(datadir, workdir, infile, outfile):
    """Collect parameters of csv files in the DATADIR listed in project file from WORKDIR"""
    getparams_fn(datadir, workdir, infile, outfile)


# STAGE1' PREPROCESS
def preprocess_fn(datadir, workdir, infile, outfile, seed):
    if outfile is None:
        # try to remove .ancinf from infile
        position = infile.find('.ancinf')
        if position > 0:
            outfile = infile[:position] + '.explist'
        else:
            outfile = infile + '.explist'
    rng = np.random.default_rng(seed)
    start = time.time()
    sim.preprocess(datadir, workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time() - start:.2f}s")


@cli.command()
@click.argument("datadir")
@click.argument("workdir")
@click.option("--infile", default="project.ancinf", help="Project file, defaults to project.ancinf")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project "
                                              "file with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def preprocess(datadir, workdir, infile, outfile, seed):
    """Filter datsets from DATADIR, generate train-val-test splits and experiment list file in WORKDIR"""
    preprocess_fn(datadir, workdir, infile, outfile, seed)


# STAGE 2 SIMULATE
def simulate_fn(workdir, infile, outfile, seed):
    if outfile is None:
        # try to remove .ancinf from infile
        position = infile.find('.params')
        if position > 0:
            outfile = infile[:position] + '.explist'
        else:
            outfile = infile + '.explist'

    rng = np.random.default_rng(seed)
    start = time.time()
    sim.simulateandsave(workdir, infile, outfile, rng)
    print(f"Finished! Total {time.time() - start:.2f}s")


@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.params", help="File with simulation parameters, defaults to project.params")
@click.option("--outfile", default=None, help="Output file with experiment list, defaults to project file "
                                              "with '.explist' extension")
@click.option("--seed", default=2023, help="Random seed")
def simulate(workdir, infile, outfile, seed):
    """Generate ibd graphs, corresponding slpits and experiment list file for parameters in INFILE"""
    simulate_fn(workdir, infile, outfile, seed)


# STAGE3 HEURISTICS GNN etc
def copyclassifiers(resds, exp):
    # existing experiment, find it
    for res_exp in resds:
        if res_exp["exp_idx"] == exp["exp_idx"]:
            break
    res_exp["dataset_time"] += exp["dataset_time"]
    # now list classifiers and add split scores
    # classifiers should be the same
    for classifier in exp["classifiers"]:
        for metric in exp["classifiers"][classifier]:
            if metric != "class_scores":
                newvals = exp["classifiers"][classifier][metric]["values"]
                res_exp["classifiers"][classifier][metric]["values"].extend(newvals)
            else:
                for pop in exp["classifiers"][classifier]["class_scores"]:
                    newvals = exp["classifiers"][classifier][metric][pop]["values"]
                    res_exp["classifiers"][classifier][metric][pop]["values"].extend(newvals)


def recomputemeanstd(exp):
    for classifier in exp["classifiers"]:
        for metric in exp["classifiers"][classifier]:
            metricresults = exp["classifiers"][classifier][metric]
            if metric != "class_scores":
                metricresults["mean"] = np.average(metricresults["values"])
                metricresults["std"] = np.std(metricresults["values"])
                if "clean_mean" in metricresults:
                    metricresults["clean_mean"] = np.average(metricresults["clean_values"])
                    metricresults["clean_std"] = np.std(metricresults["clean_values"])
            else:
                for cl in metricresults:
                    metricresults[cl]["mean"] = np.average(metricresults[cl]["values"])
                    metricresults[cl]["std"] = np.std(metricresults[cl]["values"])


def combine_splits(partresults):
    result = {}
    for partres in partresults:
        # include values from partresults
        for dataset in partres:
            if dataset in result:
                # existing dataset. list experiments and find new
                existing_exp_ids = [exp["exp_idx"] for exp in result[dataset]]
                for exp in partres[dataset]:
                    if exp["exp_idx"] in existing_exp_ids:
                        copyclassifiers(result[dataset], exp)
                    else:
                        # new experiment
                        result[dataset].append(exp)
                        result[dataset][-1]["dataset_begin"] = "multiprocessing"
                        result[dataset][-1]["dataset_end"] = "multiprocessing"
            else:
                # new dataset
                result[dataset] = partres[dataset]
                for exp in result[dataset]:
                    exp["dataset_begin"] = "multiprocessing"
                    exp["dataset_end"] = "multiprocessing"

    # recompute mean and std
    for dataset in result:
        for exp in result[dataset]:
            recomputemeanstd(exp)

    return {"brief": sim.getbrief(result), "details": result}


def runandsavewrapper(args):
    return sim.runandsaveall(args["workdir"], args["infile"], args["outfilebase"], args["fromexp"], args["toexp"],
                             args["fromsplit"], args["tosplit"], args["gpu"])


# STAGE5 TEST HEURISTICS, COMMUNITY DETECTIONS AND TRAIN&TEST NNs
def crossval_fn(workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount):
    if outfile is None:
        # try to remove .ancinf from infile
        position = infile.find('.explist')
        if position > 0:
            outfilebase = infile[:position]
        else:
            outfilebase = infile
    else:
        outfilebase = outfile

    start = time.time()
    if processes == 1:
        sim.runandsaveall(workdir, infile, outfilebase, fromexp, toexp, fromsplit, tosplit, gpu)
    else:
        # get every process only one job computing splitrange aforehead
        splitcount = int(tosplit) - int(fromsplit)
        splitsperproc = splitcount // processes
        splitincrements = [splitsperproc] * processes
        for idx in range(splitcount % processes):
            splitincrements[idx] += 1
        splitrange = [int(fromsplit)]
        incr = 0
        for idx in range(processes):
            incr += splitincrements[idx]
            splitrange.append(int(fromsplit) + incr)

        print("Split seprarators:", splitrange)

        taskargs = [{"workdir": workdir,
                     "infile": infile,
                     "outfilebase": outfilebase,
                     "fromexp": fromexp,
                     "toexp": toexp,
                     "fromsplit": splitrange[procnum],
                     "tosplit": splitrange[procnum + 1],
                     "gpu": procnum % gpucount} for procnum in range(processes)]
        print(taskargs)

        with Pool(processes) as p:
            resfiles = p.map(runandsavewrapper, taskargs)

        # now combine results
        if (fromexp is None) and (toexp is None):
            outfile_exp_postfix = ""
        else:
            outfile_exp_postfix = "_e" + str(fromexp) + "-" + str(toexp)
        outfile_split_postfix = "_s" + str(fromsplit) + "-" + str(tosplit)
        outfilename = outfilebase + outfile_exp_postfix + outfile_split_postfix + '.results'
        partresults = []
        for partresultfile in resfiles:
            with open(partresultfile, "r") as f:
                partresults.append(json.load(f)["details"])
        combined_results = combine_splits(partresults)

        with open(os.path.join(workdir, outfilename), "w", encoding="utf-8") as f:
            json.dump(combined_results, f, indent=4, sort_keys=True)

    print(f"Finished! Total {time.time() - start:.2f}s.")


@cli.command()
@click.argument("workdir")
@click.option("--infile", default="project.explist", help="File with experiment list, defaults to project.explist")
@click.option("--outfile", default=None, help="File with classification metrics, defaults to project file "
                                              "with '.result' extension")
@click.option("--seed", default=2023, help="Random seed")
@click.option("--processes", default=1, help="Number of parallel workers")
@click.option("--fromexp", default=None, help="The first experiment to run")
@click.option("--toexp", default=None, help="Last experiment (not included)")
@click.option("--fromsplit", default=None, help="The first split to run")
@click.option("--tosplit", default=None, help="Last split (not included)")
@click.option("--gpu", default=0, help="GPU")
@click.option("--gpucount", default=1, help="GPU count")
def crossval(workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount):
    """Run crossvalidation for classifiers including heuristics, community detections, GNNs and MLP networks"""
    crossval_fn(workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount)


# STAGE 4 INFERENCE
def infer_fn(workdir, infile, inferdf):
    result = sim.inference(workdir, infile, inferdf)
    outfilename = inferdf + ".inferred"

    with open(os.path.join(workdir, outfilename), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, sort_keys=True)


@cli.command()
@click.argument("workdir")
@click.argument("infile")
@click.argument("inferdf")
def infer(workdir, infile, inferdf):
    """
    Classify unknow nodes from INFERDF using all models listed in INFILE
    for which crossvalidation was already performed.

    WORKDIR: folder with .explist file and run results

    INFILE:  .explist file with a list of models for inference

    INFERDF: dataset with nodes with classes to be inferred (labelled 'unknown')
    """
    infer_fn(workdir, infile, inferdf)


# UTILS
def combine_fn(workdir, outfile):
    resfiles = sorted(glob.glob(os.path.join(workdir, "*.results")))
    partresults = []
    for partresultfile in resfiles:
        with open(partresultfile, "r") as f:
            partresults.append(json.load(f)["details"])
    combined_results = combine_splits(partresults)

    with open(os.path.join(workdir, outfile), "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4, sort_keys=True)


@cli.command()
@click.argument("workdir")
@click.argument("outfile")
def combine(workdir, outfile):
    '''
    Combine .results files from a folder
    '''
    combine_fn(workdir, outfile)


@cli.command()
@click.argument("datadir", type=click.Path(exists=True, readable=True))
def stats(datadir):
    """
    Analyzes the dataset of a graph and outputs various statistics.

    This function performs the following steps:
    1. Checks if the dataset has duplicate edges.
    2. Check if the dataset includes every region data id.
    3. Outputs various statistics about the graph, including:
       - Number of edges in the graph.
       - Number of vertices (nodes) in the graph.
       - Descriptive statistics of the graph's edge weights, including:
         - minimum
         - maximum
         - mean
         - standard deviation
         - 25th percentile
         - median
         - 75th percentile

    Parameters:
    - datadir (str): The file path to the dataset containing the graph data.

    Returns:
    None. Prints the statistics.
    """
    data_stats(datadir)


def main():
    cli()
