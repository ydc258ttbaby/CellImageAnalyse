% image = Image;
close all
filePath = "D:\武汉\胸腹腔数据\第三次\209116\209116\output95\";
srcnamelist = dir(strcat(filePath,'*.mat'));
count = 1
% for i = 1
    filename = strcat(filePath,srcnamelist(i).name);
    load (filename)
    L = length(data);
    total = round(L/1800000);

    close all
    for start = 0:total-1
        if((start+1)*round(L/total)>L)
            break;
        end
        if(start > 3)
            break;
        end
        start = 4
        partdata = data((start*round(L/total)+1):1:(start+1)*round(L/total));
        cropdata = CropData(-partdata,1000,500,0.4);
        image = ImageRecoveryModify(cropdata,2,1,0.5,1.0);

    %     figure
    %     imagesc(image)
    %     colormap(gray)
        [row,col,dep] = size(image);
        cropImg = imcrop(image, [0,230,col,100]);

        ImageRes = imresize(cropImg,[8*row col]);
        ImageCrop = f_imgCrop(ImageRes,32,328*100/row,6);
        ImageNor = f_imgNormalize(ImageCrop);
        Image = imresize(ImageNor,[500 500]);
        imwrite(Image,strcat("C:\test\",'WHc_209116_',num2str(count),'.png'));
        count = count + 1
    %     figure
    %     imshow(ImageNor)
    end        

% end
