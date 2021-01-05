clear all;
close all;
clc;
tic
j=2180;
filePath = 'D:\580\split\';
filenamelist = dir(filePath);
srcPath = filePath;
srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
LengthOfNamelist = length(srcnamelist);
for i = 2180

    disp(strcat(num2str(i),'/',num2str(LengthOfNamelist)))
    filename = strcat(srcPath,srcnamelist(i).name);
    resStruct(j).name = srcnamelist(i).name;
    if(strfind(filename,'.Wfm.bin')>0)
        file = fopen(filename,'rb');
        [data,n] = fread(file,'float32');
        pause(0.001)
        fclose(file);
    end
%             partdata = partdata(30:end);
                
                RawData = data;
                SampleRate = 200000;
                PulPol  =1;
                widthPar=0.5;
                passbandPar=1;
                
                if size(RawData, 1) > 1
                    RawData = RawData * PulPol;
                else
                    RawData = RawData' * PulPol;
                end

                % Get the Fourier transform and repetition rate of the data
                RawData = RawData - mean(RawData);
                RawData_f = fft(RawData);
            %     RepRate = find(RawData_f == max(RawData_f(100 : round(end / 2))), 1, 'first');
                RepRate = find(RawData_f == max(RawData_f(100 : 4500)), 1, 'first');


                % Calculate the width of each pulse and make sure the number is even
                Width = round(widthPar * length(RawData) / RepRate) * 2;


                % Smooth the signal by low pass filtering 
                PassBand = round(passbandPar * RepRate);
                filter = [0; ones(PassBand, 1); zeros(length(RawData) - 1 - 2 * PassBand, 1); ones(PassBand, 1)];
                FilterData = real(ifft(RawData_f .* filter));


                % Calculate the start point of first pulse
                SubData = FilterData(1 : Width * 4);
                % [Peaks, Locs] = findpeaks(SubData); %不使用findpeaks
                Locs = find(diff(sign(diff(SubData)))==-2)+1;
                if Locs(1) - Width / 2 > 0
                    FirstPulsePos = Locs(1) - Width / 2;
                else
                    FirstPulsePos = Locs(2) - Width / 2;
                end
                FirstPulse = RawData(FirstPulsePos : FirstPulsePos + Width);


                % Locate the last pulse by cross correlation
                CrossCor = [];
                for i = length(RawData)- Width * 4 : length(RawData)- Width
                    CrossCor = [CrossCor, sum(FirstPulse .* RawData(i : i + Width))];
                end
                LastPulsePos = find(CrossCor == max(CrossCor), 1, 'last') + length(RawData)- Width * 4 - 1;

                % Calculate the number of pulses between the first and last pulses (number of column in the image)
                FilterData = FilterData(FirstPulsePos : LastPulsePos - 1);
                % ColNum = length(findpeaks(FilterData)) + 1; %不使用findpeaks
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

                % Construct the 2D image from raw data
                Image = RawData(ImgIndex);
                Image = reshape(Image, Width + 1, []);

                % Get rid of the pulse profile
                Background1 = mean(Image(:, 5 : 25), 2);
                Background2 = mean(Image(:, end - 25 : end-5), 2);
                if mean(Background1) >= mean(Background2)
                    Background = Background1;
                else
                    Background = Background2;
                end
                Image = Image - repmat(Background, 1, ColNum);
        
    resStruct(j).RepRate = RepRate;
    resStruct(j).Width = Width;
    [row,col,dep] = size(Image);
    imagesc(Image);
    colormap('gray');

%             set(h2,'position',[100 100 1200 720]);
    [x y]=ginput();
    x = round(x);
    y = round(y);
    if(length(x) ==2)
        point = x * row;
        resStruct(j).x1 = x(1);
        resStruct(j).x2 = x(2);
        resStruct(j).y1 = y(1);
        resStruct(j).y2 = y(2);
        figure
        plot_1D_Single(data,'data')
        hold on
        plot(point(1):1:point(2),ones(1,length(point(1):1:point(2)))*0.03,'linewidth',3);
        pause(0.5);
    else
        resStruct(j).x1 = 0;
        resStruct(j).x2 = 0;
        resStruct(j).y1 = 0;
        resStruct(j).y2 = 0;
    end
    GrayImage = ImageNormalize(Image);
    RSGrayImage = imresize(GrayImage,[row col*0.35]);
    [n,m,l] = size(RSGrayImage);
    %RSGrayImage = imcrop(RSGrayImage, round([1, n/4, m, n/2]));
    imwrite(RSGrayImage,strcat('D:\图像数据\580forDL\','580_',num2str(j),'.png'));
                
    j=j+1

    close all;
    clear y;
    clear x;
    
end
toc
% resCell = struct2cell(resStruct);
% writecell(resCell,'C.csv')
beep


