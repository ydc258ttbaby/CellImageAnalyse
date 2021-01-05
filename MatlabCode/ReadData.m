function [data,row,col] = ReadData(filename)
    
    if(strfind(filename,'.bin')>0)
        file = fopen(filename,'rb');
        [data,num] = fread(file,'float32');
        %data = data(3:end,:);% 此处为了去除文件头
        [row,col] = size(data);
        pause(0.001)
        fclose(file);
    end
    filename = "D:\天津\csv数据\19-10-30_10_1.Wfm.csv";
    if(strfind(filename,'.csv')>0)
        data = readmatrix(filename);
        [row,col] = size(data);
    end
end