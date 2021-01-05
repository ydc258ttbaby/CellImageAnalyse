function res = f_imgNormalize(src)
% F_IMGNORMALIZE normalize the image
%   RES = F_IMGNORMALIZE(SRC) map the image SRC from [imgMin,imgMax] to
%   [0,255] and convert format to UINT8
%
    norMax = 255;
    norMin = 0;
    imgMax = max(max(src)); 
    imgMin = min(min(src)); 
    res = uint8(round((norMax-norMin)*(src-imgMin)/(imgMax-imgMin) + norMin));
end