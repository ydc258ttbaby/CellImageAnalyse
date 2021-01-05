clear all;
close all;
clc;
tic
j=1;
binName = '323_';
filePath = 'D:\580\';
desPath = 'D:\580\split\';
% desPath = "C:\test\"
filenamelist = dir(filePath);
srcPath = filePath;
srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
LengthOfNamelist = length(srcnamelist);
for i = 1:LengthOfNamelist

    disp(strcat(num2str(i),'/',num2str(LengthOfNamelist)))
    filename = strcat(srcPath,srcnamelist(i).name);
%     filename = "Z:\RTx\RefWaveforms\wave20201124_T110907_80_5.Wfm.bin";
    if(strfind(filename,'.Wfm.bin')>0)
        file = fopen(filename,'rb');
        [data,n] = fread(file,'float32');
        pause(0.001)
        fclose(file);
    end
    
    L = length(data);
    rawdata = data;
%     total = 80;
    total = getNumFromName(srcnamelist(i).name);
%         total = 1;
    for start = 0:total-1;
        partdata = data((start*round(L/total)+1):1:(start+1)*round(L/total));
%             partdata = partdata(30:end);
        fid=fopen(strcat(desPath,binName,num2str(j),'.Wfm.bin'),'wb');
        j=j+1;
        fwrite(fid,partdata,'float32');
        fclose(fid);
    end
end
toc
beep


