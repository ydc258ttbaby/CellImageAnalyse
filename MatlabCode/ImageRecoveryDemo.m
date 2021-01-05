% clear all
% filepathname = "E:\RefWaveforms\2020-12-08_50_064247.Wfm.bin";
% % filepathname = "C:\ydc\学习\实验室\天津院\示波器数据\Data\goodRes\wave1120201014_T141404.Wfm.bin";
% if(strfind(filepathname,'.Wfm.bin')>0)
%     file = fopen(filepathname,'rb');
%     [data,n] = fread(file,'float32');
%     pause(0.001)
%     fclose(file);
% end
% data = data(40:end);
% close all;
% filename = "E:\清华\原始数据\208879\100001.bin";
% [data] = f_readData(filename);
% data = data(42:end);
close all
load D:\胸腹腔数据\第二次\331\data7
figure
plot_1D_Single(data,'data')

RawData = data(end-2000000:end);
figure
plot_1D_Single(RawData,'RawData')
% RawData = cropData;
% RawData = partdata;
SampleRate = 1000000;
PulPol =1;
display =1;
tic
widthPar = 0.5
passbandPar = 1.0
% for widthPar = 0.5
%     for passbandPar = 1.0
        if display==1
        figure
        plot_1D_Single(RawData(1:4000),'RawData')
        end
        % Make sure the pulses are positive and RawData is a column vector
        if size(RawData, 1) > 1
            RawData = RawData * PulPol;
        else
            RawData = RawData' * PulPol;
        end

        % Get the Fourier transform and repetition rate of the data
        RawData = RawData - mean(RawData);
        
        RawData_f = fft(RawData);
        
        if display ==1
        figure
        plot_1D_Single(abs(RawData_f(1:100000)),'RawData_f')
        end
        RepRate = find(RawData_f == max(RawData_f(1 :30000)), 1, 'first');
%         RepRate = 2000;

        % Calculate the width of each pulse and make sure the number is even
        Width = round(widthPar * length(RawData) / RepRate) * 2;


        % Smooth the signal by low pass filtering 
        PassBand = round(passbandPar * RepRate);
%         PassBand = 2500
        filterArray = [ones(1900,1); ones(PassBand-1899, 1); zeros(length(RawData) - 1 - 2 * PassBand, 1); zeros(PassBand, 1)];
        if display == 1
            figure
            plot_1D_Single(filterArray(1:10000),'filter')
        end
        filterArrayHighPass = 1-filterArray;
        PassBand = 0
        FilterDataHigh = real(ifft(RawData_f .* [0; zeros(PassBand, 1); ones(length(RawData) - 1 - 2 * PassBand, 1); ones(PassBand, 1)]));
        if display==1
        figure
        plot_1D_Single(FilterDataHigh(1:5000),'FilterData')
        end
        FilterData = real(ifft(RawData_f .* filterArray));
        FilterDataRaw = FilterData;
        if display==1
        figure
        plot_1D_Single(FilterData(1:5000),'FilterData')
        end
%         FilterData = smoothdata(RawData,'gaussian',400);
        
        if display==1
        figure
        plot_1D_Single(FilterData(1:end),'FilterData')
        end

        % Calculate the start point of first pulse
        SubData = FilterData(1 : Width * 4);
        % [Peaks, Locs] = findpeaks(SubData); %不使用findpeaks
        Locs = find(diff(sign(diff(SubData)))==-2)+1;
        TotalLocs = find(diff(sign(diff(FilterData)))==-2)+1;
        if Locs(1) - Width / 2 > 0
            FirstPulsePos = Locs(1) - Width / 2;
        else
            FirstPulsePos = Locs(2) - Width / 2;
        end
        FirstPulse = RawData(FirstPulsePos : FirstPulsePos + Width);
%         figure
%         plot_1D_Single(FilterData(length(FilterData)-4*Width:end),'FirstPulse')

        % Locate the last pulse by cross correlation
        CrossCor = [];
        for i = length(RawData)- Width * 4 : length(RawData)- Width
            CrossCor = [CrossCor, sum(FirstPulse .* RawData(i : i + Width))];
        end
        LastPulsePos = find(CrossCor == max(CrossCor), 1, 'last') + length(RawData)- Width * 4 - 1;
%         LastPulsePos = 799086
        % Calculate the number of pulses between the first and last pulses (number of column in the image)
        FilterData = FilterData(FirstPulsePos : LastPulsePos - 1);
        % ColNum = length(findpeaks(FilterData)) + 1; %不使用findpeaks
        ColNum = length(find(diff(sign(diff(FilterData)))==-2)) + 1;


        % Calculate the exaxt duration of the pulses (should be non-integer)
        LastPulsePos = LastPulsePos
        Duration = (LastPulsePos - FirstPulsePos) / (ColNum - 1);

        % Calculate the indexs of the first elements in each column
        StartPoint = round((0 : ColNum - 1) * Duration) + FirstPulsePos - 1 ;
        StartPoint2 = (TotalLocs(1:end-1))' - 200;
%         if display == 1
%             figure
%             plot_1D_Single(smoothdata(diff(StartPoint2),'gaussian',40),'StartPoint2')
%         end
        ImgIndex = reshape(repmat((0 : Width)', 1, ColNum) + repmat(StartPoint, Width + 1, 1), 1, []);

        %fixing bugs when sometime starpoint will get the value zero //testing
        if ImgIndex(1) == 0
            ImgIndex(1)=1;
        end

        % Construct the 2D image from raw data
        Image = RawData(ImgIndex);
        Image = reshape(Image, Width + 1, []);
        if display==1
        figure
        mesh(Image)
        title('-bg before')
        end
        % Get rid of the pulse profile
        Background1 = mean(Image(:, 50 : 250), 2);
        Background2 = mean(Image(:, end - 250 : end-50), 2);
        if mean(Background1) >= mean(Background2)
            Background = Background1;
        else
            Background = Background2;
        end
        if display == 1
            
            figure
            imagesc(Image);
            colormap(gray);
            title(strcat(num2str(widthPar),'-',num2str(passbandPar)));
        end
        Image = Image - repmat(Background, 1, ColNum);
        mean(mean(Image))-min(min(Image))
        if display == 1
            figure
            mesh(repmat(Background, 1, ColNum));
            title('bg')
            figure
            imagesc(Image);
            colormap(gray);
            figure
            mesh(Image)
            title(strcat(num2str(widthPar),'-',num2str(passbandPar)));
        end
%         RepRate = (RepRate - 1) * SampleRate / length(RawData);
%     end
% end
toc
% sqrt(std2(Image))/mean(mean(Image))