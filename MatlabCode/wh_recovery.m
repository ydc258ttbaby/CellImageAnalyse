% image = Image;
close all
filePath = "E:\胸腹腔数据（此为武汉原始数据）\第二次\331\";
srcnamelist = dir(strcat(filePath,'*.mat'));
count = 19
for i = 8:length(srcnamelist)
    filename = strcat(filePath,srcnamelist(i).name);
    load (filename)
    L = length(data);
    total = round(L/1800000);

    close all
    for start = 0:total-1
        if((start+1)*round(L/total)>L)
            break;
        end
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
        imwrite(Image,strcat("G:\武汉\第二次图像数据\原始数据\331\32um\",'WH_331_',num2str(count),'.png'));
        count = count + 1
    %     figure
    %     imshow(ImageNor)
    end        

end
