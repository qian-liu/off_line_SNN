function net = cnntrain(net, x, y, opts, af)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    net.L = [];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        % Decaying learning rate
        if mod(i-1,2)== 0 && i>1 %2
            opts.alpha = opts.alpha*0.9;
        end
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            net = cnnff(net, batch_x, opts, af);
            net = cnnbp(net, batch_y, opts, af);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L(end);
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L(end);
            
            disp(sprintf('numbatch: %d, RL: %.3f, loss: %.3f, out: %.3f, 3a: %.4f, 4a: %.4f, 5a: %.4f',...
                l, net.rL(end), net.L(end),...
                mean(mean(net.o)),...
                mean(mean(mean(net.layers{3}.a{1}))),... %', 4k: ' num2str(net.layers{4}.k{1}{1}(1)),...
                mean(mean(mean(net.layers{4}.a{1}))),... %', 5d: ' num2str(net.layers{5}.d{1}(1)),...
                mean(mean(mean(net.layers{5}.a{1})))));
        end
        toc;
    end
    
end
