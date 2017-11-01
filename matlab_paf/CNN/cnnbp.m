function net = cnnbp(net, y, opts, af)
    n = numel(net.layers);

    %   error
    net.e = net.o - y;
    %  loss function
    net.L(end+1) = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
    
    p = af.tau_syn * af.S;    
    if strcmp(af.name, 'ReLU')
        paf_der = @(x)double(x>0) * p; 
    elseif strcmp(af.name, 'Noisy_Softplus')
        paf_der = @(x, sigma, af)der_noisy_softplus(x, sigma, af);
    end
    
    
    %%  backprop deltas = d L/d y * d y/d net
    if strcmp(af.name, 'Noisy_Softplus') %  output delta
        net.od = net.e .* paf_der(net.net, net.noise_o, af);
    else
        net.od = net.e .* paf_der(net.net);
    end
    
    net.fvd = (net.ffW' * net.od);              %  feature vector delta, backprop 1 layer
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        if strcmp(af.name, 'Noisy_Softplus') %  output delta
            net.fvd = net.fvd .* paf_der(net.fv, net.noise_fv, af);
        else
            net.fvd = net.fvd .* paf_der(net.fv);
        end
    end

    %  reshape feature vector deltas into output map style
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                d_lup = expand(net.layers{l + 1}.d{j}, ...
                            [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) ...
                            / (net.layers{l + 1}.scale^2 * p); 
                            
                            
                if strcmp(af.name, 'Noisy_Softplus') %  output delta
                    net.layers{l}.d{j} = paf_der(net.layers{l}.net{j}, net.layers{l}.noise{j}, af) .* d_lup;
                else
                    net.layers{l}.d{j} = paf_der(net.layers{l}.net{j}) .* d_lup;
                end
                
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                
                net.layers{l}.d{i} = z;
%                 if strcmp(af.name, 'Noisy_Softplus') %  output delta
%                     net.layers{l}.d{i} = z .* paf_der(net.layers{l}.net{i}, net.layers{l}.noise{i}, af); %%please comment here!!!
%                 else
%                     net.layers{l}.d{i} = z .* paf_der(net.layers{l}.net{i}); %%please comment here!!!
%                 end
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                if opts.bias
                    net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
                end
            end
        end
    end
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    if opts.bias
        net.dffb = mean(net.od, 2);
    end
    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
