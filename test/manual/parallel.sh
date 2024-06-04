python3 -m ancinf preprocess ../datasets/ ./test5/ --infile simple.ancinf
python3 -m ancinf crossval ./test5 --infile simple.explist --processes 2 --gpucount 2 --fromsplit 0 --tosplit 2
