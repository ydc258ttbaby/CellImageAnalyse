close all;
RawData = data;
display=1;
if display == 1
    figure
   plot_1D_Single(RawData(1:3000),'Rawdata 1-3000'); 
end
[SmoothData,w] = smoothdata(RawData,'gaussian',400);
w
if display == 1
    figure
   plot_1D_Single(SmoothData(1:3000),'SmoothData 1-3000'); 
end

RawData = RawData - mean(RawData);
RawData_f = fft(RawData);
if display ==1
figure
plot_1D_Single(abs(RawData_f(1:50000)),'RawData_f')
end
RepRate = find(RawData_f == max(RawData_f(100 : 4500)), 1, 'first');


% Calculate the width of each pulse and make sure the number is even
% Width = round(widthPar * length(RawData) / RepRate) * 2;

passbandPar = 1;
% Smooth the signal by low pass filtering 
PassBand = round(passbandPar * RepRate);
filter = [0; ones(PassBand, 1); zeros(length(RawData) - 1 - 2 * PassBand, 1); ones(PassBand, 1)];
FilterData = real(ifft(RawData_f .* filter));
if display==1
figure
plot_1D_Single(FilterData(1:3000),'FilterData')
end