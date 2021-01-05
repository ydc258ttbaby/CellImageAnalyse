clc;
clear;

DPO = visa('ni', ['TCPIP::', '192.168.1.1', '::INSTR'], 'InputBufferSize', 255e6);
fopen(DPO);
out=query(DPO,'ACQuire:RESolution?');
a=str2num(out);
SampleRate = 1/a;
SampleRate = 1 / str2num(query(DPO,'ACQuire:RESolution?'));

% fprintf(DPO,'FORMat:DATA INT,16');
% fprintf(DPO,'FORM ASC');
% fprintf(DPO,'EXP:WAV:INCX OFF');
% fprintf(DPO,'RUNSingle');
count = 1;

% status = str2num(query(DPO,'STATus:OPERation:CONDition?'));
% while status == 312
%     pause(0.5);
%     status = str2num(query(DPO,'STATus:OPERation:CONDition?'));
% end

RawData = str2num(query(DPO,'CHAN1:WAV1:DATA?'));
plot_1D_Single(RawData,'raw');
pause(0.05);
disp('ydc')
%[Image, RepRate] = ImageRecovery(RawData, SampleRate, 1);

%figure(1);
%imagesc(Image);

fprintf(DPO,'RUNSingle');

