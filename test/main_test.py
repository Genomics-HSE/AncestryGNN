import ancinf

def test_helook():
    # python3 -m ancinf preprocess ../datasets/ ./test5/ --infile simple.ancinf
    ancinf.preprocess_fn("./test/datasets/", "./test/manual/test5/", "simple.ancinf", None, 2023)
    # python3 -m ancinf crossval ./test5/ --infile simple.explist 
    # workdir, infile, outfile, seed, processes, fromexp, toexp, fromsplit, tosplit, gpu, gpucount
    ancinf.crossval_fn("./test/manual/test5/", "simple.explist", None, 2023, 1, None, None, None, None, 0, 1)
    assert True 