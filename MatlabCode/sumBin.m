close all;
clc;

binName = '576_'
desPath = "D:\天津\原始数据\576\split\"
filePath = "D:\天津\原始数据\576\";
filenamelist = dir(filePath);
j = 1;
for h = 1:length(filenamelist)
    srcPath = strcat(filePath,filenamelist(h).name,'\');
    srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
    LengthOfNamelist = length(srcnamelist);
    for i = 1:LengthOfNamelist
        copyfile(strcat(srcPath,srcnamelist(i).name),strcat(desPath,binName,num2str(j),'.Wfm.bin'),'f');
        j=j+1;
    end
end