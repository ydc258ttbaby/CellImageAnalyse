close all
tic
srcPath = 'D:\图像数据\胸腔积液\576\rename\';
desPath = 'D:\图像数据\胸腔积液\576\rgb\'
imgName = '576_';
srcnamelist = dir(strcat(srcPath,'*.jpg'));
for i = 1:length(srcnamelist)
    disp(strcat(num2str(i),'/',num2str(length(srcnamelist))))
    filename = strcat(srcPath,srcnamelist(i).name);
    src = imread(filename);
%     figure
%     imshow(src)
%     [mx,my] = ginput();
%     src = rgb2gray(src);
    src = src(200:2299,900:2699,:);
%     src = imcrop(src, round([750,200, 2900-800, 2300-200]));
    n = 2100;
    m = 1800;
%     des = rgb2gray(src);
    
    a = 1499;
    wid = a+1;
    filterWid = 300;
    newImage = uint8(ones(n+wid,m+wid,3).*mean(mean(src(1:100,1:2099))));
    newImage(wid/2+1:wid/2+n,wid/2+1:wid/2+m,:) = src;
%     figure
%     imshow(newImage)
    I_3=fspecial('average',[filterWid,filterWid]);%3*3均值滤波
    des=imfilter(newImage,I_3);
    des = des(filterWid/2+1:n+wid-filterWid/2,filterWid/2+1:m+wid-filterWid/2);
%     figure
%     imshow(des)
    [y,x] = find(des==min(min(des)),1,'first');
    
    sx = min(n,max(1,x-a/2));
    sy = min(m,max(1,y-a/2));
    cropImg = imcrop(newImage, round([sx+filterWid/2,sy+filterWid/2, a, a]));
%     figure
    imshow(cropImg)
%     cropImg = gray2rgb(cropImg);
    
%     RSGrayImage = imresize(cropImg,[a a]);
%                 
    imwrite(cropImg,strcat(desPath,imgName,num2str(i),'.png'));
    
end
toc