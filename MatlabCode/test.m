% image = Image;
close all
filePath = "F:\胸腹腔数据\第一次\576\output19(576)\";
srcnamelist = dir(strcat(filePath,'*.mat'));
count = 2101
for i = 1:length(srcnamelist)
    filename = strcat(filePath,srcnamelist(i).name);
    load (filename)
    L = length(data);
    total = round(L/1800000);

    close all
    for start = 0:total-1
        partdata = data((start*round(L/total)+1):1:(start+1)*round(L/total));
        image = ImageRecoveryModify(-partdata,2,1,0.5,1.0);

    %     figure
    %     imagesc(image)
    %     colormap(gray)
        [row,col,dep] = size(image);
        cropImg = imcrop(image, [0,230,col,100]);

        ImageRes = imresize(cropImg,[8*row col]);
        ImageCrop = f_imgCrop(ImageRes,32,328*100/row,6);
        ImageNor = f_imgNormalize(ImageCrop);
        Image = imresize(ImageNor,[500 500]);
        imwrite(Image,strcat("D:\武汉\第一次图像数据\576\recovery\",'WH_576_',num2str(count),'.png'));
        count = count + 1
    %     figure
    %     imshow(ImageNor)
    end        

end
