import ancinf

def test_helook():
    #python3 -m ancinf preprocess ../datasets/ ./test5/ --infile simple.ancinf
    ancinf.preprocess_fn("./test/datasets/", "./test/manual/test5/", "simple.ancinf", None, 2023)
    #print('dir(ancinf)')
    assert True 