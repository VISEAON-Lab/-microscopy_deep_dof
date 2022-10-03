function [finalKernel, add, rows] = perFocusPlaneKernel(path, numerical_factor, x)
        load(path);
        res_f = (Dx/5.5) * numerical_factor; 
        rows_aux = ceil(x * res_f);
        k_res = imresize(y2(1:end-1, 1:end-1), [rows_aux rows_aux]);

        [rows, cols] = size(k_res);
        if rows ~= cols 
            error(path);
        end

        [X, Y] = meshgrid(1:rows, 1:rows);
        [Xq, Yq] = meshgrid(1:0.5:rows, 1:0.5:rows);
        Vq = interp2(X,Y,k_res,Xq,Yq);
        
        add = x/2 - rows/2; %calculating the shift
        finalKernel = Vq(1+ceil(mod(add,1)):2:end, 1+ceil(mod(add,1)):2:end);
        finalKernel = finalKernel / sum(finalKernel(:));
end