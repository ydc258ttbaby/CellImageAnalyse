
srcPath = 'Z:\RTx\RefWaveforms\';
desPath = 'C:\ydc\学习\实验室\天津院\RawData\';
srcnamelist = dir(strcat(srcPath,'*.bin'));
LengthOfNamelist = length(srcnamelist);
for i = 1:LengthOfNamelist
    movefile(strcat(srcPath,srcnamelist(i).name),strcat(desPath,srcnamelist(i).name),'f');
end
