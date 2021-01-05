
%clear all;
close all;
if exist('DPO','var')~=0
    fclose(DPO);
    delete(DPO)
    clear DPO
end
isControlSerial = 1;
if isControlSerial
    if exist('s','var')==0
        s=serialport('com7',115200);%初始化串口
        configureTerminator(s,"CR")
    end
    writeline(s,"run");
end
acNumGoal = 160;

m = zeros(1,10);
tic

for j = 1:999
    tic
    DPO = visa('ni', ['TCPIP::', '192.168.1.1', '::INSTR'], 'InputBufferSize', 255e6);
    fopen(DPO);
    fprintf(DPO,'EXPort:WAVeform:RAW OFF');
    fprintf(DPO,'EXPort:WAVeform:INCXvalues OFF');
    fprintf(DPO,'EXPort:WAVeform:FASTexport ON');
    fprintf(DPO,'CHANnel1:WAVeform1:STATe 1');
    fprintf(DPO,'EXPort:WAVeform:SOURce C1W1');
    fprintf(DPO,'EXPort:WAVeform:SCOPe WFM');
    fprintf(DPO,'EXPort:WAVeform:DLOGging ON');
    fprintf(DPO,strcat('ACQuire:COUNt',32,num2str(acNumGoal)));

%     status = str2num(query(DPO,'STATus:OPERation:CONDition?'));
%     while status == 312
%         pause(0.5);
%         status = str2num(query(DPO,'STATus:OPERation:CONDition?'));
%     end
    curtime = datestr(now,'yyyymmdd THHMMSS');
    curtime(find(isspace(curtime))) = '_';

    filename = strcat('wave',curtime,'_',num2str(acNumGoal),'_',num2str(j),'.Wfm.bin');
    fprintf(DPO,strcat('EXPort:WAVeform:NAME ''D:\tianjin\wave',curtime,'_',num2str(acNumGoal),'_',num2str(j),'.bin'''));
    fprintf(DPO,strcat('EXPort:WAVeform:NAME ''C:\Users\Public\Documents\Rohde-Schwarz\RTx\RefWaveforms\wave',curtime,'_',num2str(acNumGoal),'_',num2str(j),'.bin'''));
    
    disp('等待触发完成')
    query(DPO,'RUNSingle');
    c = 1;
    while(1)
        ac = query(DPO,'ACQuire:AVAilable?');
%             if(c > 50)
%                 fprintf(DPO,'TRIGger:FORCe')
%                 c=1;
%             end
        if(str2num(ac) == acNumGoal)
            break;
        else
            %pause(0.01);
            c = c + 1;
        end
    end
    disp('采集')
    t1 = toc
    
    if(t1 < 13 || t1 > 150)
        beep
    end
    disp('pause')
    if isControlSerial
        writeline(s,"stop");
    end
    pause(14)
%     test
    disp('传输+处理+存储')
    t2 = toc - t1
    if isControlSerial
        writeline(s,"run");
    end
    disp('end')
    fclose(DPO);
    delete(DPO)
    clear DPO
end

toc
if isControlSerial
    writeline(s,"stop");
end


