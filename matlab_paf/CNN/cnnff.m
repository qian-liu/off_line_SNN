function net = cnnff(net, x, opts, af)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    p = af.tau_syn * af.S;    
    if strcmp(af.name, 'ReLU')
        paf = @(x)max(0,x) * p; 
    elseif strcmp(af.name, 'Noisy_Softplus')
        paf = @(x, sigma, af)noisy_softplus(x, sigma, af);
        %paf = @(x, sigma, af)((-10<=x & x<=10 & sigma~=0).*af.nsp_k.*sigma.*log(1+exp(x./(af.nsp_k.*sigma)))...
        %                      + (x<-10 | x>10 | sigma==0).*max(0,x)) * p; 
        %paf = @(x, sigma, af)af.nsp_k.*sigma.*log(1+exp(x./(af.nsp_k.*sigma)))*p;
    end
    
    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                % add bias
                if opts.bias
                    z = z + net.layers{l}.b{j};
                end
                
                % Noisy_Softplus required another dimontion as input: noise
                net.layers{l}.net{j} = z;
                if strcmp(af.name, 'Noisy_Softplus')
                    noise_z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                    for i = 1 : inputmaps   %  for each input map
                        %  convolve with corresponding kernel and add to temp output map
                        noise_z = noise_z + convn(abs(net.layers{l - 1}.a{i}), 0.5*net.layers{l}.k{i}{j}.^2, 'valid');
                    end
                    noise_z = sqrt(noise_z);
                    % pass through activation funciton
                    net.layers{l}.a{j} = paf(z,noise_z,af);
                    net.layers{l}.noise{j} = noise_z;
                else
                    net.layers{l}.a{j} = paf(z);
                end
                
                
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                w = ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2) / p; %averaging pooling
                z = convn(net.layers{l - 1}.a{j}, w, 'valid');  
                net.layers{l}.net{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                if strcmp(af.name, 'Noisy_Softplus')
                    noise_z = sqrt(convn(abs(net.layers{l - 1}.a{j}), 0.5 * w.^2, 'valid'));
                    z = paf(z,noise_z,af);
                    net.layers{l}.noise{j} = noise_z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
                else
                    z = paf(z);
                end
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    z = net.ffW * net.fv;
    if opts.bias
        z = z + repmat(net.ffb, 1, size(net.fv, 2));
    end
    net.net = z;  
    
    if strcmp(af.name, 'Noisy_Softplus')
        for j = 1 : numel(net.layers{n}.a)
            net.noise_fv = [];
            net.noise_fv = [net.noise_fv; reshape(net.layers{n}.noise{j}, sa(1) * sa(2), sa(3))];
        end
        noise_z = sqrt(0.5 * net.ffW.^2* abs(net.fv));
        net.o = paf(z,noise_z,af);
        net.noise_o = noise_z;
    else
        net.o = paf(z);
    end    

end
