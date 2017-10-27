cnn_file=Noisy_Softplus_e20_r4_acc98.66_s.mat #ReLU_e20_r4_acc99.07_s.mat
newdir=${cnn_file::${#cnn_file}-4}
matdir=../results/$newdir #name the result folder
logfile=../results/$newdir/$newdir.txt  #name the logfile
mkdir $matdir

#config_file=../../matlab_paf/model/default_config.mat
af_file=../../matlab_paf/model/Noisy_Softplus #ReLU.mat
python scnn_test_NS.py $af_file ../../matlab_paf/results/$cnn_file #>>$logfile
python read_results.py ../results/$newdir/ 10000 100,400,1000
