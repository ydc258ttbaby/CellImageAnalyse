clear all;
close all;
clc;


csvdata = importdata("C:\ydc\pythonCode\cells_label_res.csv");
L = length(csvdata.textdata)
for i = 1:L
    filename = strcat("D:\天津\原始数据\580\split\",csvdata.textdata(i));
    if(strfind(filename,'.Wfm.bin')>0)
        file = fopen(filename,'rb');
        [data,n] = fread(file,'float32');
        pause(0.001)
        fclose(file);
    end
    preData = csvdata.data(2*i-1,:);
    realData = csvdata.data(2*i,:);
    
    [Image, RepRate] = ImageRecoveryModify(data, 1,1,0.5,1);
    [row,col,dep] = size(Image);
    
    imagesc(Image)
    colormap('gray')
    
    %[x,y]=ginput()
    rectangle('Position',[round(col*(realData(1)-realData(3)/2)),round(row*(realData(2)-realData(4)/2)),round(col*realData(3)),round(row*realData(4))],'EdgeColor','g')
    rectangle('Position',[round(col*(preData(1)-preData(3)/2)),round(row*(preData(2)-preData(4)/2)),round(col*preData(3)),round(row*preData(4))],'EdgeColor','r')
    pause(0.5)
end
