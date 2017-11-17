# off_line_SNN
off-line training of SNN in Matlab and SNN testing in PyNN lauguage (still under tidy-up).   
The first version of SNN trainig is written in Matlab, which is a modification of the DeepLearnToolbox (https://github.com/rasmusbergpalm/DeepLearnToolbox).

## Usage of Matlab_PAF
1. in your command window go to the dir Matlab_PAF
2. run the script file:  
**script**
3. you can change the default configurations of the training, such as **num_epoch**, of the matfile in the models folder.
4. the training outcomes (the weights and etc.) will be saved in the results folder.


## Usage of PyNN + Nest
The version I am using is PyNN 0.7.5 + Nest 2.2.2, it will probably not working for other combinations of these tools.  
1. To run an example go to the code_snn_nest folder, and run   
**./sh_test.sh**    
2. You can also analysis the results with different number of testing images, e.g. first 100 images, and various length of testing times, such as 100,400,1000 ms  
**python read_results.py ../results/ReLU_ft 100 100,400,1000**
