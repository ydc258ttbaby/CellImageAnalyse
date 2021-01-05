widthPar = 1.0;
passbandPar = 0.7;

close all;
rawData = data;
figure
plot_1D_Single(rawData(200:1000),'rawData 200:1000')

SampleRate = 20000000;
PulPol = 1;

% Make sure the pulses are positive and RawData is a column vector
if size(rawData, 1) > 1
    rawData = rawData * PulPol;
else
    rawData = rawData' * PulPol;
end

% Get the Fourier transform and repetition rate of the rawData
rawData = rawData - mean(rawData);
RawData_f = fft(rawData);
RepRate = find(RawData_f == max(RawData_f(1 : round(end / 2))), 1, 'first');


% Calculate the width of each pulse and make sure the number is even
Width = round(widthPar * length(rawData) / RepRate) * 2;



% Smooth the signal by low pass filtering 
PassBand = round(passbandPar * RepRate);
filter = [0; ones(PassBand, 1); zeros(length(rawData) - 1 - 2 * PassBand, 1); ones(PassBand, 1)];
FilterData = real(ifft(RawData_f .* filter));
figure
plot_1D_Single(FilterData(200:1000),'filterdata 200:1000')


% Calculate the start point of first pulse
SubData = FilterData(1 : Width * 4);
%[Peaks, Locs] = findpeaks(SubData);
Locs = find(diff(sign(diff(SubData)))==-2)+1;
if Locs(1) - Width / 2 > 0
    FirstPulsePos = Locs(1) - Width / 2;
else
    FirstPulsePos = Locs(2) - Width / 2;
end
FirstPulse = rawData(FirstPulsePos : FirstPulsePos + Width);


% Locate the last pulse by cross correlation
CrossCor = [];
for i = length(rawData)- Width * 4 : length(rawData)- Width
    CrossCor = [CrossCor, sum(FirstPulse .* rawData(i : i + Width))];
end
LastPulsePos = find(CrossCor == max(CrossCor), 1, 'last') + length(rawData)- Width * 4 - 1;

% Calculate the number of pulses between the first and last pulses (number of column in the image)
FilterData = FilterData(FirstPulsePos : LastPulsePos - 1);
%ColNum = length(findpeaks(FilterData)) + 1;
ColNum = length(find(diff(sign(diff(FilterData)))==-2)) + 1;


% Calculate the exaxt duration of the pulses (should be non-integer)
Duration = (LastPulsePos - FirstPulsePos) / (ColNum - 1);

% Calculate the indexs of the first elements in each column
StartPoint = round((0 : ColNum - 1) * Duration) + FirstPulsePos - 1;
ImgIndex = reshape(repmat((0 : Width)', 1, ColNum) + repmat(StartPoint, Width + 1, 1), 1, []);

%fixing bugs when sometime starpoint will get the value zero //testing
if ImgIndex(1) == 0
    ImgIndex(1)=1;
end

% Construct the 2D image from raw rawData
Image = rawData(ImgIndex);
Image = reshape(Image, Width + 1, []);

% Get rid of the pulse profile
figure
mesh(Image);
title('img and BG')
Background1 = mean(Image(:, 5 : 25), 2);
Background2 = mean(Image(:, end - 25 : end-5), 2);
if mean(Background1) >= mean(Background2)
    Background = Background1;
else
    Background = Background2;
end
BG = repmat(Background, 1, ColNum);
figure
mesh(BG);
title('BG');
Image = Image - repmat(Background, 1, ColNum);
%RepRate = (RepRate - 1) * SampleRate / length(RawData);

figure
mesh(Image);
title('mesh img')
figure
imagesc(Image);
title('img')