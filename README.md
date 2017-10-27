# off_line_SNN
off-line training of SNN in Matlab and SNN testing in PyNN lauguage (still under tidy-up).

## Usage of Matlab_PAF
1. in your command window please add the path first:  
**addpath(genpath('../matlab_paf'));**  
2. run the CNN example:  
**test_example_CNN('Noisy_Softplus')** or  
**test_example_CNN('ReLU')**
3. you can change the default configurations of the training, such as **num_epoch**, of the matfile in the models folder.
4. the training outcomes (the weights and etc.) will be saved in the results folder.
