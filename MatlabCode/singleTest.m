filepathname = "C:\ydc\学习\实验室\天津院\示波器数据\Data\goodRes\wave1120201014_T141404.Wfm.bin";
if(strfind(filepathname,'.Wfm.bin')>0)
    file = fopen(filepathname,'rb');
    [data,n] = fread(file,'float32');
    pause(0.001)
    fclose(file);
end
data = data(40:end);
[Image, RepRate] = ImageRecoveryModify(data, 20,1,0.5,1.0);
figure
imagesc(Image);
colormap(gray);