close all
tic;
total = 1;
rawdata = data;
L = length(rawdata);

isDisplay = 1;
for start =0
    
% data = rawdata((start*round(L/total)+1):1:(start+1)*round(L/total));
lengthOfData = length(data);
if isDisplay
    figure
    plot_1D_Single(data,'data')
end
window_size = 400;
filter_size = 100;
alpha = 0.4;
[row,col] =size(data);
subinterval_num = floor(row/window_size);
subinterval_max_list =zeros(subinterval_num,1);
maxValue = 0;
smoothData = zeros(subinterval_num,1);
minV = +inf;
maxV = -inf;
sumV = 0;
leftIndex = 1;
rightVal = 0;
count = 0;
filter_sum = 0;
isExistSingle = 1;

for i = 1:subinterval_num
    index = ((i-1)*window_size+1):(i*window_size);
    % 平均分为多段数据，每段取最大值
    maxValue = max(data(index));
    subinterval_max_list(i) = maxValue;
    % 对每段数据的最大值构成的新数列进行平滑
    filter_sum = filter_sum + maxValue; 
    if(i > filter_size)
       filter_sum = filter_sum - subinterval_max_list(i-filter_size);
    end
    tempData = filter_sum/min(i,filter_size);
    smoothData(i) = tempData;
    % 同时保存最大值和最小值
    if(i>filter_size)
    minV = min(minV,tempData);
    maxV = max(maxV,tempData);
    sumV = sumV + tempData;
    end
end
meanV = sumV / (subinterval_num-filter_size);
dropV = max(maxV - meanV,meanV-minV);
smoothData = smoothData(filter_size:end);
if isDisplay
    figure
    plot_1D_Single(smoothData,'SmoothData')
end
upRange = find(smoothData > meanV + alpha*dropV);
downRange = find(smoothData < meanV - alpha*dropV);
if isDisplay
    hold on 
    plot(1:length(smoothData),ones(1,length(smoothData))*(meanV + alpha*dropV));
    plot(1:length(smoothData),ones(1,length(smoothData))*(meanV - alpha*dropV));
    plot(upRange,ones(1,length(upRange))*(meanV + alpha*dropV),'linewidth',5);
    plot(downRange,ones(1,length(downRange))*(meanV - alpha*dropV),'linewidth',5);
end
if ~isempty(upRange) && ~isempty(downRange)
   leftRange = min(upRange(1),downRange(1));
   rightRange = max(upRange(end),downRange(end));
end
if isempty(upRange) && ~isempty(downRange)
   leftRange = downRange(1);
   rightRange = downRange(end);
end
if ~isempty(upRange) && isempty(downRange)
   leftRange = upRange(1);
   rightRange = upRange(end);
end
if isempty(upRange) && isempty(downRange)
   isExistSingle = 0; 
end
if isExistSingle
    range = rightRange-leftRange;
    centerIndex = floor((rightRange+leftRange)/2+filter_size/2);
    trueRange = window_size * range;
    cropRange = 2^(length(dec2bin(trueRange))-1)*2;
    if(cropRange < lengthOfData/2)
        cropIndex = (centerIndex*window_size - cropRange + 1):(centerIndex*window_size +cropRange);
        cropData = data(cropIndex);
    else
        cropRange = 2^(length(dec2bin(lengthOfData/2))-1);
        cropIndex = (floor(cropRange+10000/2) - cropRange + 1):(floor(cropRange+10000/2) +cropRange);
        cropData = data(cropIndex);
    end
    [Image, RepRate] = ImageRecoveryModify(cropData, 20000000,1,0.5,1.0);
    
    if isDisplay
        figure
        plot_1D_Single(data,'cropData');
        hold on 
        plot(cropIndex,ones(1,length(cropIndex))*meanV,'linewidth',3);
        h = figure
        set(h,'position',[100 100 600*(cropRange*2/lengthOfData) 360]);
        imagesc(Image);
        colormap(gray);
        [x y]=ginput()
        disp('ydc')
    end
%     GrayImage = ImageNormalize(Image);
%     RSGrayImage = imresize(GrayImage,[401 668*(cropRange*2/lengthOfData)]);
%     [n,m,l] = size(RSGrayImage);
%     des = imcrop(RSGrayImage, round([1, n/4, m, n/2]));
%     imwrite(des,strcat('C:\ydc\学习\实验室\天津院\',num2str(start),'.png'));
else
    disp('not found single');
    cropData = [];
end
end
toc
