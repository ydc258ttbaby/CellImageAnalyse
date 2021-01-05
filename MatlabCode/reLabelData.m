% clear all;
close all;
clc;


csvdata = importdata("C:\ydc\pythonCode\580LabelBBOX.csv");
L = length(csvdata.textdata)
i=1;
while(i<=L)
    i
    filename = strcat("D:\天津\原始数据\580\split\",csvdata.textdata(i));
    resStruct(i).name = char(csvdata.textdata(i));
    if(strfind(filename,'.Wfm.bin')>0)
        file = fopen(filename,'rb');
        [data,n] = fread(file,'float32');
        pause(0.001)
        fclose(file);
    end
%     preData = csvdata.data(2*i-1,:);
    realData = csvdata.data(i,:);
    
    [Image, RepRate] = ImageRecoveryModify(data, 1,1,0.5,1);
    [row,col,dep] = size(Image);
    
    imagesc(Image)
    colormap('gray')
    
    %[x,y]=ginput()
    rectangle('Position',[round(col*(realData(1)-realData(3)/2)),round(row*(realData(2)-realData(4)/2)),round(col*realData(3)),round(row*realData(4))],'EdgeColor','g')
%     rectangle('Position',[round(col*(preData(1)-preData(3)/2)),round(row*(preData(2)-preData(4)/2)),round(col*preData(3)),round(row*preData(4))],'EdgeColor','r')
    Mess = input('wait');
    if(Mess == 0)
        i=i-1;
        continue;
    end
    if(~isempty(Mess))
        [x y]=ginput();
        x = x/col;
        y = y/row;
        if(length(x) ==2)
            resStruct(i).x = (x(1)+x(2))/2;
            resStruct(i).y = (y(1)+y(2))/2;
            resStruct(i).w = x(2)-x(1);
            resStruct(i).h = y(2)-y(1);
            resStruct(i).isNoise = 0;
        else
            resStruct(i).x = 0;
            resStruct(i).y = 0;
            resStruct(i).w = 0;
            resStruct(i).h = 0;
            resStruct(i).isNoise = 1;
        end
    else
        resStruct(i).x = realData(1);
        resStruct(i).y = realData(2);
        resStruct(i).w = realData(3);
        resStruct(i).h = realData(4);
        resStruct(i).isNoise = 0;
    end
    close all;
    clear y;
    clear x;
    i=i+1;
        
end
resCell = struct2cell(resStruct);
writecell(resCell,'C.csv')