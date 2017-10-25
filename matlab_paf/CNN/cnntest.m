function [er, bad] = cnntest(net, x, y, opts, af)
    %  feedforward
    net = cnnff(net, x, opts, af);
    [~, h] = max(net.o);
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end
