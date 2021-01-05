% clear all
close all
tic

display = 1


filePathList = ["D:\天津\图像数据\209116\"];
for k = 1:length(filePathList)
    filePath = filePathList(k);
%     filePath = "E:\天津\图像数据\";
    desPath1 = strcat(filePath,'crop18\');
    mkdir(desPath1);
    desPath2 = strcat(filePath,'crop25\');
    mkdir(desPath2);
    srcnamelist = dir(strcat(filePath,'*.png'));
    centerList = zeros(2,length(srcnamelist));
    for i = length(srcnamelist)
        disp(strcat(num2str(i),'/',num2str(length(srcnamelist)),'___',filePath))
        filename = strcat(filePath,srcnamelist(i).name);

        src = imread(filename);
%         src=rgb2gray(src);
        [m,n,l] = size(src);

        rowData = src(5,:);
        width = n/2;
        avgRow = ones(1,n)*mean(rowData);
        totalResData = zeros(1,n);
        for j = 1:6
            resData = (smoothdata(rowData,'gaussian',width)-ones(1,n)*mean(rowData));
            rowData = double(rowData) - resData;
            totalResData = totalResData + resData;
            width = max(width/1.4,m/5);
        end

        resMatrix = (repmat(totalResData,m,1));
        resImg = uint8(double(src) - resMatrix);


        RawImg = resImg;
        avg = mean(mean(RawImg(1:20,:)));
    %     RawImg = rgb2gray(RawImg);
        [m,n,l] = size(RawImg);
        if display == 1
            figure
            imshow(RawImg)
        end
        cropRangeR = round(m/2);
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
        
        dSigma =6.5;
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
        centerList(1,i) = cr;
        centerList(2,i) = cc;
        cropRangeR = round(m*18/56);
        cropRangeC = cropRangeR;
        cropImg = imcrop(extendImg, round([cc-cropRangeC/2,cr-cropRangeR/2, cropRangeC, cropRangeR]));
        if display == 1
            figure
            imshow(uint8(cropImg))
        end
        cropImg = imresize(cropImg,[400,400]);
        resImage = zeros(400,400,3);
        for k = 1:3
           resImage(:,:,k) =  cropImg;
        end
        resImage = uint8(resImage);
        imwrite(uint8(resImage),strcat(desPath1,'crop18_',srcnamelist(i).name));
        
        
        cropRangeR = round(m*25/56);
        cropRangeC = cropRangeR;
        cropImg = imcrop(extendImg, round([cc-cropRangeC/2,cr-cropRangeR/2, cropRangeC, cropRangeR]));
        if display == 1
            figure
            imshow(uint8(cropImg))
        end
        cropImg = imresize(cropImg,[400,400]);
        resImage = zeros(400,400,3);
        for k = 1:3
           resImage(:,:,k) =  cropImg;
        end
        resImage = uint8(resImage);
        imwrite(uint8(resImage),strcat(desPath2,'crop25_',srcnamelist(i).name));

    %     figure
    %     imshow(src)

    end
end