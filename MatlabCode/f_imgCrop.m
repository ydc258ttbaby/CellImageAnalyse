function cropImg = f_imgCrop(RawImage,crop_H,full_H,dSigma)

    display = 0;
    RawImg = RawImage;
    avg = mean(mean(RawImg));
%     RawImg = rgb2gray(RawImg);
    [m,n,l] = size(RawImg);
    if display == 1
        figure
        imshow(RawImg)
    end
    cropRangeR = round(m/4);
    cropRangeC = cropRangeR;
    extendImg = ones(m+cropRangeR,n+cropRangeC)*avg;
    extendImg(round(cropRangeR/2)+1:round(cropRangeR/2)+m,round(cropRangeC/2)+1:round(cropRangeC/2)+n) = RawImg;
    if display == 1
        figure
        imagesc(extendImg)
    end
    smallSize = 100;
    smallImg = imresize(extendImg,[smallSize,smallSize]);

    ImgMatrix = sqrt(sqrt(abs(smallImg - avg)));
%     ImgMatrix = min(ImgMatrix,ones(smallSize,smallSize)*0.9*max(max(ImgMatrix)));
    if display == 1
        figure
        imagesc(ImgMatrix)
    end
    leftTopR = round(cropRangeR/m/2*smallSize);
    leftTopC = leftTopR;
    halfRangeR = round(leftTopR/2);
    halfRangeC = round(leftTopC/2);
    [sm,sn,~] = size(smallImg); 
    sumMatrix = zeros(sm,sn);

%     dSigma =6.5;
    fK1=1.0/(2*dSigma*dSigma);
    fK2=fK1/pi;
    iSize = halfRangeR*2+1;
    out = zeros(1,iSize);
    step = floor(iSize/2 + 0.5);
    for j = 1 : iSize
        x=j-step;
        fTemp=fK2*exp(-x*x*fK1);
        out(1,x+step) = fTemp;
    end
    dM = sum(out);
    model = out / dM;
    if display == 1
        figure
        plot_1D_Single(model,'model')
    end

    for r = leftTopR:(sm-leftTopR+1)
        for c = leftTopC:(sn-leftTopC+1)
            sumMatrix(r,c) = sum(sum(model'*model.*ImgMatrix(r-halfRangeR:r+halfRangeR,c-halfRangeC:c+halfRangeC)));
        end
    end

    if display == 1
        figure
        imagesc(sumMatrix)
    end
    [cr,cc] = find(sumMatrix==max(max(sumMatrix)));

    cr = mean(cr);
    cc = mean(cc);
    if display == 1
        hold on
        plot(cc,cr,'*r')
    end
    [em,en,~] = size(extendImg);
%         cr = cr - 0.5;
%         cc = cc - 0.5;
    cr = round(cr * (em/smallSize));
    cc = round(cc * (en/smallSize));
    
    
    newCropRangeR = round(m*crop_H/full_H);
    newCropRangeC = newCropRangeR;
    cr = max(min(cr,round(m+cropRangeR/2-newCropRangeR/2)),round(cropRangeR/2+newCropRangeR/2));
    cc = max(min(cc,round(n+cropRangeC/2-newCropRangeC/2)),round(cropRangeC/2+newCropRangeC/2));
    cropImg = imcrop(extendImg, round([cc-newCropRangeC/2,cr-newCropRangeR/2, newCropRangeC, newCropRangeR]));
    if display == 1
        figure
        imshow(uint8(cropImg))
    end
%     cropImg = uint8(cropImg);
end