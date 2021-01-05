
%% 判断路径是否存在，若不存在则创建
filePath = "C:\test\";
if exist(filePath,'dir')==0
    mkdir(filePath);
end 

%% 判断fileName的文件格式，若为 .Wfm.bin 则读取二进制文件至 data

if(strfind(fileName,'.Wfm.bin')>0)
    file = fopen(fileName,'rb');
    [data,n] = fread(file,'float32');
    pause(0.001)
    fclose(file);
end

%%