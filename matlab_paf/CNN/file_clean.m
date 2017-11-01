function [ output_net ] = file_clean( input_net )
%FILE_CLEAN Summary of this function goes here
%   Detailed explanation goes here
    output_net.layers = cell(size(input_net.layers));
    for l = 1 : numel(input_net.layers)
        if input_net.layers{l}.type == 'i'
            output_net.layers{l} = input_net.layers{l};
        elseif input_net.layers{l}.type == 'c'
            output_net.layers{l}.type = input_net.layers{l}.type;
            output_net.layers{l}.outputmaps = input_net.layers{l}.outputmaps;
            output_net.layers{l}.kernelsize = input_net.layers{l}.kernelsize;
            output_net.layers{l}.k = input_net.layers{l}.k;
        elseif input_net.layers{l}.type == 's'
            output_net.layers{l}.type = input_net.layers{l}.type;
            output_net.layers{l}.scale = input_net.layers{l}.scale;
        end
    end
    
    output_net.ffW = input_net.ffW;
    output_net.rL = input_net.rL;

end

