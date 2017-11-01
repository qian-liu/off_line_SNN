function net = cnnapplygrads(net, opts)
    for l = 2 : numel(net.layers)
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for ii = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha(l) * net.layers{l}.dk{ii}{j};
                    net.layers{l}.k{ii}{j} = min(net.layers{l}.k{ii}{j},opts.max_w); 
                    net.layers{l}.k{ii}{j} = max(net.layers{l}.k{ii}{j},-opts.max_w); 
                end
                if opts.bias
                    net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha(l)  * net.layers{l}.db{j};
                end
            end
        end
    end

    net.ffW = net.ffW - opts.alpha(numel(net.layers)) * net.dffW;
    if opts.bias
        net.ffb = net.ffb - opts.alpha(numel(net.layers)) * net.dffb;
    end
end
