function [image] = f_imgRemoveDCStripe(src,debug)
% F_IMGREMOVEDCSTRIPE remove the veritcal stripes for Direct Current (DC)
%   [IMAGE] = F_IMGREMOVEDCSTRIPE(SRC,DEBUG) removes the vertical stripes 
%   caused by the high DC value in the grayimage SRC, DEBUG controls if the
%   debugger information is displayed.
% 
%   Copyright THU_EE_YDC.
    [m,n,l] = size(src);
    rowData = src(5,:);
    width = n/2;
    totalResData = zeros(1,n);
    for i = 1:6
        resData = (smoothdata(rowData,'gaussian',width)-ones(1,n)*mean(rowData));
        rowData = double(rowData) - resData;
        totalResData = totalResData + resData;
        width = max(width/1.4,m/5);
    end
    
    resMatrix = (repmat(totalResData,m,1));
    image = uint8(double(src) - resMatrix);
end
