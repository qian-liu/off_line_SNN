function test_fine_tune( trained_file, af_file, config_file, result_file)
%TEST_FINE_TUNE Summary of this function goes here
%   parameters:
%     trained_file: name of the input model file
%     af_file: activation function parameters e.g. ReLU_config.mat
%     config_file: parmeters for network training, such as network
%       architecture and number of epochs. e.g. small_config
%     result_file: name of the output file containing the finetuned weights

    
    %loading
    load mnist_uint8; %database MNIST
    load(af_file); %parameters for the Parametric Activation Function (PAF)
    load(config_file); %parameters for CNN training
    opts.alpha = af.alpha; %learning rate
    
    train_x = double(reshape(train_x',28,28,60000))/255; %training images
    test_x = double(reshape(test_x',28,28,10000))/255; %testing images
    train_y = double(train_y') + opts.offset; %training label
    test_y = double(test_y'); %testing label
    
    %equivalent input abstract K=200Hz and tau_syn=0.005
    train_x =  train_x * af.K * af.tau_syn;
    test_x = test_x * af.K * af.tau_syn;
    
    %% ex1 Train a 6c-2s-12c-2s Convolutional neural network as default
    rand('state',opts.randseed); %set random seed
    
    load(trained_file);
    
    cnn = cnntrain(cnn, train_x, train_y, opts, af);

    [er, bad] = cnntest(cnn, test_x, test_y, opts, af);
    fprintf('Testing Accuracy: %2.2f%%.\n', (1-er)*100);
    
    %tidy up fileds of cnn to be saved
    cnn = file_clean(cnn);
    cnn.acc = (1-er)*100;
    cnn.randseed = opts.randseed;
    cnn.numepochs =  opts.numepochs;
    cnn.af_file = af_file;
    cnn.config_file = config_file;
    
    %result file name
    fname = sprintf('results/%s.mat', result_file);
    save(fname, 'cnn');
    
    %plot mean squared error  
    figure; plot(cnn.rL);
end

