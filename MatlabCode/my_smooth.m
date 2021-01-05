function [smooth_data,minV,maxV] = my_smooth(data,windows)
    len = length(data);
    num = floor(len/windows);
    smooth_data = data;
    minV = +inf;
    maxV = -inf;
    for i = 1:len
        minI = max(1,i-floor(num/2));
        maxI = min(len,i+floor(num/2));
        temp = mean(data(minI:maxI));
        smooth_data(i) = temp;
        minV = min(temp,minV);
        maxV = max(temp,maxV);
    end
    
end