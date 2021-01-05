function plot_1D_Single(data,str)
    [m,n] = size(data);
    if(m>1)
        x = 1:m;
        plot(x,data)
    else if(n > 1)
        x = 1:n;
        plot(x,data)
        else
            disp('m <= 1 && n <= 1')
        end
    end 
    title(str)
end