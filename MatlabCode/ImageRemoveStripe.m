clear all
close all
tic
filePath ="E:\TEST\";
srcnamelist = dir(strcat(filePath,'*.png'));

for i = 1:length(srcnamelist)
    disp(strcat(num2str(i),'/',num2str(length(srcnamelist))))
    filename = strcat(filePath,srcnamelist(i).name);
    src = imread(filename);
    [m,n,l] = size(src);
    
    rowData = src(5,:);
    width = n/2;
    avgRow = ones(1,n)*mean(rowData);
    totalResData = zeros(1,n);
    for i = 1:6
        resData = (smoothdata(rowData,'gaussian',width)-ones(1,n)*mean(rowData));
        rowData = double(rowData) - resData;
        totalResData = totalResData + resData;
        width = max(width/1.4,m/5);
    end
    
    resMatrix = (repmat(totalResData,m,1));
    resImg = uint8(double(src) - resMatrix);
    imwrite(resImg,filename);
end
toc