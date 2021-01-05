clc;clear all;close all;

srcPath = 'Z:\RTx\RefWaveforms\';
curtime = datestr(now,'yyyymmdd THHMMSS');
curtime(find(isspace(curtime))) = '_';
desPath = strcat('C:\ydc\学习\实验室\天津院\RawData\data_',curtime);
if exist(desPath,'dir')~=0
    rmdir(desPath, 's')
end   
mkdir(desPath);
holdOffTime = 0;
totalNum = 0;
while(holdOffTime < 10)
    startTime = clock;
    srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
    % 每次从示波器取文件的时候，都留一个不取，所以判定 LengthOfNamelist>1 说明有新文件，代表触发
    LengthOfNamelist = length(srcnamelist);
    for i = 1:LengthOfNamelist-1
        movefile(strcat(srcPath,srcnamelist(i).name),strcat(desPath,'\',srcnamelist(i).name),'f');
        totalNum = totalNum + 1;
        disp(strcat('成功转移  ',num2str(totalNum),'个数据'));
        % holdOffTime代表这次触发和上次触发的时间差，lastTriggerTime代表上次触发时间点
        if exist('lastTriggerTime','var')
            holdOffTime = etime(clock,lastTriggerTime);
            disp(strcat('距上次转移相隔：',num2str(holdOffTime),'s'));
        end
        lastTriggerTime = clock;
        
    end
    % 图片的显示规则
    % LengthOfNamelist > 1代表是否触发,delayTime代表距离上次显示图片经过的时间
    if((exist('delayTime','var')~=0 && ( LengthOfNamelist > 3))|| (exist('delayTime','var')==0 &&  LengthOfNamelist > 3))
        filename = strcat(srcPath,srcnamelist(1).name);
        filename = 'Z:\RTx\RefWaveforms\2020-09-24_3_081754.Wfm.bin';
        file = fopen(filename,'rb');
        [RawData,n] = fread(file,'float32');
        max(RawData)
        RawData = RawData(1:end,:);% 此处为了去除文件头
        close all;
        fig = figure('NumberTitle', 'off', 'Name', filename);
        plot_1D_Single(RawData,'RawData');
        pause(0.001)
        fclose(file);
        if exist('lastDisplayTime','var')
            delayTime = etime(clock,lastDisplayTime);
            disp(strcat('距上次显示相隔：',num2str(delayTime),'s'));
            disp('--------------------')
        end
        lastDisplayTime = clock;
    end
end