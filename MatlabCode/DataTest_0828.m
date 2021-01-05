% fileNameBIN = 'C:\ydc\学习\实验室\天津院\数据\a00314.bin';
% [dataBIN,m,n] = ReadData(fileNameBIN);
% 
% fileNameCSV = 'C:\Users\86133\Documents\selfData\test.csv';
% [dataCSV,m,n] = ReadData(fileNameCSV);
% dataCSV = dataCSV(:,2);

clear all;
close all
filename = 'C:\ydc\学习\实验室\天津院\示波器数据\Data\RefWaveforms\goodRes\wave101.Wfm.bin';
[dataCSV2,row,col] = ReadData(filename);
L = length(dataCSV2);
figure
plot_1D_Single(dataCSV2,'RawData')
start = 9;
num = 10;
i=36;
data = dataCSV2(start*round(L/num)+1:1:(start+1)*round(L/num));
%data=data(50:end-50);
SampleRate = 20000000;

%cropData = DataCrop(data,1000,20,0.5);
cropData = data;
[Image, RepRate] = ImageRecoveryModify(cropData, SampleRate,1);
figure
imagesc(Image);
print('-dpng',strcat('p_',num2str(i),'.png'));
%plot_1D_Single(cropData,'cropIndex')
% figure
% ImageNorm = Image/max(max(Image));
% imshow(ImageNorm)