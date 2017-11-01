source ~/env/nest_pynn0.7/bin/activate

cnn_file=ReLU_small_ft.mat #trained weights
newdir=${cnn_file::${#cnn_file}-4}
matdir=../results/$newdir #name the result folder
logfile=../results/$newdir/$newdir.txt  #name the logfile
mkdir $matdir


af_file=../../matlab_paf/model/ReLU_config

# Test the trained weights on SNN simulations on Nest
python scnn_test_NS.py $af_file ../../matlab_paf/results/$cnn_file >>$logfile
# Analysis output spikes to evluate the recognition accuracy with during test of 100ms, 400ms, 1s respectively
python read_results.py ../results/$newdir/ 10000 100,400,1000
