function Image = ImageRecoveryTHU(RawData)
    format long e;
    ss=500;                                   %与一个周期内点数有关
    delta=[-2:1:2];                           %ss值可能不固定，允许有-2~+2误差

    mdata=RawData;                    %第二列代表电压
    maxp=max(mdata);                      %读取极值
    minp=min(mdata);
    th=0.16*(maxp-minp)+minp;             %设定阈值二值化
    data=double(mdata>th);
    %% 调整波形至中央
    datast=[mdata(1:ss);mdata(1:ss);mdata(1:ss)];   %三倍扩展第一个周期数据，便于找到波形中央
    cha1=zeros(ss,1);
    for j=1:ss
        cha1(j)=sum(datast(ss+j-12:ss+j+12));     %求相邻24个数据的和，便于找到最低点
    end
    [x,y]=min(cha1);                             %找最小值位置
    mdata=mdata(y:end);                          %数据开始保证是下凹线，波形即在中间
    data=data(y:end);
    %% 波形对齐
    data_len=length(data);
    len=round(data_len/ss);                      %数据周期长度
    dtdq=data(1:ss);                             %第一个周期波形
    zhizhen=ss+1;
    Image=zeros(ss,len+30);                      %初始化一个二维矩阵
    Image(:,1)=mdata(1:ss);
    for j=2:len+15
        cha2=zeros(5,1);
        for jj=1:5
            zhizhenj=zhizhen+delta(jj);
            dataj=data(zhizhenj:zhizhenj+ss-1);
            cha2(jj)=sum(sum((dataj-dtdq).^2));   %利用最小二乘法对齐
        end
        [x,y]=min(cha2);                           %重新排列数据，组成二维矩阵
        zhizhen=zhizhen+delta(y);
        Image(:,j)=mdata(zhizhen:zhizhen+ss-1);
        zhizhen=zhizhen+ss;
        if (zhizhen+ss-1+2)>data_len
            break
        end
    end
    if j<(len+30)
        Image(:,(j+1):end)=[];
    end
    Background1 = mean(Image(:, 50 : 250), 2);
    Background2 = mean(Image(:, end - 250 : end-50), 2);
    if mean(Background1) >= mean(Background2)
        Background = Background1;
    else
        Background = Background2;
    end
    
    Image = Image - repmat(Background, 1, size(Image,2));
end

    
    