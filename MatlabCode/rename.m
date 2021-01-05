close all
tic
filePathList = [
                "D:\武汉\第三次图像数据\208894\Rename\"
%                 "D:\武汉\第三次图像数据\209172\"
                ];
for k = 1:length(filePathList)
    filePath = filePathList(k);
%     filePath = "E:\天津\图像数据\";
%     desPath1 = strcat(filePath,'Rename\');
%     mkdir(desPath1);
    srcnamelist = dir(strcat(filePath,'*.png'));
    for i = 1:length(srcnamelist)
        disp(strcat(num2str(i),'/',num2str(length(srcnamelist)),'___',filePath))
        filename = strcat(filePath,srcnamelist(i).name);
        movefile(filename,strcat(filePath,'WH_',srcnamelist(i).name))
%         src = imread(filename);
%         res = imresize(src,[500,500]);
%         imgname = srcnamelist(i).name;
%         imwrite(res,strcat(desPath1,imgname(1:end-4),'png'));
    
    end
end