function test_example_CNN(af_name)
    %adding the working path of the tool
    addpath(genpath('../../matlab_paf'));
    
    %loading
    load mnist_uint8; %database MNIST
    load(sprintf('%s.mat',af_name)); %parameters for the Parametric Activation Function (PAF)
    load('default_config.mat'); %parameters for CNN training
    opts.alpha = af.alpha; %learning rate
    
    train_x = double(reshape(train_x',28,28,60000))/255; %training images
    test_x = double(reshape(test_x',28,28,10000))/255; %testing images
    train_y = double(train_y'); %training label
    test_y = double(test_y'); %testing label
    
    %equivalent input abstract K=200Hz and tau_syn=0.005
    train_x =  train_x * af.K * af.tau_syn;
    test_x = test_x * af.K * af.tau_syn;
    %% ex1 Train a 6c-2s-12c-2s Convolutional neural network as default
    rand('state',opts.randseed); %set random seed

    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', opts.convmaps(1), 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', opts.convmaps(2), 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
    };
    

    
    cnn = cnnsetup(cnn, train_x, train_y, opts, af);
    cnn = cnntrain(cnn, train_x, train_y, opts, af);

    [er, bad] = cnntest(cnn, test_x, test_y, opts, af);
    fprintf('Testing Accuracy: %2.2f%%.\n', (1-er)*100);
    
    %result file name
    fname = sprintf('results/%s_e%d_r%d_acc%2.2f.mat', af.name, opts.numepochs, opts.randseed, (1-er)*100);
    save(fname, 'cnn');
    
    %plot mean squared error  
    figure; plot(cnn.rL);
    %assert(er<0.12, 'Too big error');
end