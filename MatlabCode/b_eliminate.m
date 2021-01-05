clear;
clc;
close all;
fclose all;
format long e;
%% 输出参数、调用文件
str0='F:\wr20201211\208894预处理\';
str1=strcat(str0,'duiqi\');
str2=strcat(str0,'back\');
load max_15;                   %上载a_adjust中处理得到的极值
mmax=max15(1);
mmin=max15(2);
bamax=-2;                      %先定义背景初始极值
bamin=1;
num=100;
scale=6;
%% 读取数据
file=dir(fullfile(str1,'*.mat'));
[n,~]=size(file);
%%
for i=1:n
    i                                      %输出便于观察进度
    mdata0=load([str1,file(i).name]);      %读取二维矩阵ss*length，包含图像信息
    strr=file(i).name(1:end-4);            %文件存储名
    data=[mdata0.datap];                   %？？没明白.datap，这一步目的为何，调整数据格式？？
    data=(data-mmin)*1000/(mmax-mmin);     %为何要乘以1000
    [ss,len]=size(data);
    fch=zeros(ss,1);
    for j=1:ss
        fch(j)=var(data(j,:));             %求每一行方差
    end
    [x,y]=max(fch);                        %方差较大值理论为细胞所在位置，y为细胞所在行数
    xst=y-num;
    xend=y+num;
    if xst<1
        xst=1;
    end
    if xend>ss
        xend=ss;
    end
    cxx=zeros(5,1);
    xxx=[1,round(len*1/5)+1,round(len*2/5)+1,round(len*3/5)+1,round(len*4/5)+1];
    yyy=[round(len*1/5),round(len*2/5),round(len*3/5),round(len*4/5),len];
    %%
    for j=1:5
        cdata=data(xst:xend,xxx(j):yyy(j));        %%分区域，xst-xend默认有细胞
        [ss_c,len_c]=size(cdata);
        len_cmid=round(len_c/2);
        cdata_1=(sum(cdata(:,len_cmid-4:len_cmid+5)')/10)';    %求中间10列平均值
        cdata=cdata-repmat(cdata_1,[1,len_c]);
        cxx(j)=std2(cdata);                          %求标准差
    end
    [x,y]=min(cxx);                                  %求标准差最小区域
    yst=xxx(y);
    yend=yyy(y);
    %% 去背景
    dataxy=data(xst:xend,yst:yend);           %提取出data的一部分区域
    [ss,len]=size(dataxy);
    xxx=[1,round(len*1/5)+1,round(len*2/5)+1,round(len*3/5)+1,round(len*4/5)+1];      %将dataxy分为5个区域
    yyy=[round(len*1/5),round(len*2/5),round(len*3/5),round(len*4/5),len];
    cxx=zeros(5,1);
    for j=1:5
        cdata=dataxy(:,xxx(j):yyy(j));
        [ss_c,len_c]=size(cdata);
        len_cmid=round(len_c/2);
        cdata_1=(sum(cdata(:,len_cmid-4:len_cmid+5)')/10)';    %求中间10列平均值
        cdata=cdata-repmat(cdata_1,[1,len_c]);
        cxx(j)=std2(cdata);                          %求标准差
    end
    [x,y]=min(cxx);                                  %提取出方差最小点所在位置
    yyst=xxx(y)+yst-1;                               %找到背景所在位置
    yyend=yyy(y)+yst-1;                              %yyend-yyst=1/25数据长度，即1/25图象长度
    yrp=round((yyst+yyend)/2);
    [ss,len]=size(data);
    cst=(sum(data(:,yrp-17:yrp+18)')/36)';           %背景
    cpp=data-repmat(cst,[1,len]);                    %去背景
    
    %% 更新极值矩阵
    maxp=max(max(cpp));
    minp=min(min(cpp));
    if bamax<maxp
        bamax=maxp;
    end
    if bamin>minp
        bamin=minp;
    end
    %% 存数据和图象
    save(strcat(str2,strr),'cpp');
    c=cpp;
    [~,lenp]=size(c);             %理论上c是ss*lenp的矩阵
    c=(c-minp)*1/(maxp-minp);
    ph=imresize(c,[ss,round(lenp/scale)],'bicubic');       %重新调整矩阵大小
    imwrite(ph,strcat(str2,'img',strr,'.png'));            %存图片
    maxback=[bamax,bamin];                                 %更新极值
    save('max_back','maxback');
    delete([str1,file(i).name]);                           %删除原始数据
end