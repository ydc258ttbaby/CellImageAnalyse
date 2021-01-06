filename = "C:\Data\QQ_files\3290707042\FileRecv\15-23-27.bin";
if(strfind(filename,'.bin')>0)
    file = fopen(filename,'rb');
    [data,num] = fread(file,'uint16');
    %data = data(3:end,:);% 此处为了去除文件头
    [row,col] = size(data);
    pause(0.001)
    fclose(file);
end
L = length(data)
figure
plot_1D_Single(data,'data')
partdata = data(1:320*512*6);
img = reshape(partdata,320,512*6);
figure
imagesc(img)
colormap('gray')
frame = f_imgNormalize(img);


aviobj = VideoWriter('driver3.avi');
aviobj.FrameRate = 10;
open(aviobj)
%我制作了由180张图片构成的视频
col = L /512/1995;
% col = col/2;
list = [89,329,569,809,1049,1289,1528,1768,123]
for i = 89
    partdata = data(col*512*(i-1)+1:col*512*i);
    img = reshape(partdata,col,512);
    img = img';
    img = flipud(img);
    img = fliplr(img);
%     figure
%     imagesc(img(10:end-10,10:end-10))
%     colormap('gray')
%     frame = img;
    frame = f_imgNormalize(img);
    frame = f_imgNormalize(img(10:end-10,10:end-10));
%     imwrite(frame,strcat('D:\研究生工作\天津院工作相关\code\CellImageAnalyse\MatlabCode\imgs2\',num2str(i),'.png'));
%     writeVideo(aviobj,frame);
%     pause(0.1)
    i
end
close(aviobj)
close all
