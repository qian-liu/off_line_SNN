function [ y ] = noisy_softplus( x, sigma, af)
%NOISY_SOFTPLUS Summary of this function goes here
%   Detailed explanation goes here
    p = af.tau_syn * af.S;
    valid_ind = find(x<10 & x>-10 & sigma~=0);
    y = max(0,x) * p;
    y(valid_ind) = af.nsp_k.*sigma(valid_ind).*log(1+exp(x(valid_ind)./(af.nsp_k.*sigma(valid_ind))))*p;

end

