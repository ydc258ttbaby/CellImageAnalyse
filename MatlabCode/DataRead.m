
close all;
clc;
tic

curtime = datestr(now,'yyyymmdd THHMMSS');
curtime(find(isspace(curtime))) = '_';
desPath = strcat('Z:\RTx\RefWaveforms\',curtime);
if exist(desPath,'dir')~=0
    rmdir(desPath, 's')
end   
mkdir(desPath);

%filename = 'Z:\RTx\RefWaveforms\2020-10-15_43_071509.Wfm.bin';
while(1)
    srcPath = 'Z:\RTx\RefWaveforms\';
    srcnamelist = dir(strcat(srcPath,'*.bin'));
    LengthOfNamelist = length(srcnamelist);
    for i = 1:LengthOfNamelist
        filename = strcat(srcPath,srcnamelist(i).name);
        if(strfind(filename,'.Wfm.bin')>0)
            file = fopen(filename,'rb');
            [data,n] = fread(file,'float32');
            pause(0.001)
            fclose(file);
        end
        if(strfind(filename,'.csv')>0)
            data = readmatrix(filename);
            [row,col] = size(data);
        end
        movefile(strcat(srcPath,srcnamelist(i).name),strcat(desPath,'\',srcnamelist(i).name),'f');

        L = length(data);
        rawdata = data;
        j=36;
        for i = 0:0

            start = i
            total = 1;

        %     figure
        %     plot_1D_Single(data,'RawData')
            data = rawdata((start*round(L/total)+1):1:(start+1)*round(L/total));
            %figure
            %plot_1D_Single(data,strcat('data',32,num2str(start),' in',num2str(total)))
            
            DataRecovery

        end

    end
end
toc