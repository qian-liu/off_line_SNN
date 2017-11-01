% add the working path of matlab
addpath(genpath('../'));
% train a CNN with ReLU, trained weights will be in 'results/ReLU_small.mat'
test_example_CNN('ReLU_config', 'small_config', 'ReLU_small');
% fine tune the CNN with Noisy Softplus
% fine tuned weights will be save in 'results/ReLU_small_ft.mat'
test_fine_tune( 'ReLU_small', 'Noisy_Softplus_config', 'fine_tune_config', 'ReLU_small_ft')