clear;
clc;
close all;
fclose all;
format long e;
%% 输出参数、调用文件
str0='F:\wr20201211\208894预处理\';
str1=strcat(str0,'back\');       %原始数据所在位置
str2=strcat(str0,'cut\');        %截图后存的位置
load max_back;                   %上b_eliminate中处理得到的极值
mmax=max15(1);
mmin=max15(2);
jies=100;                        %和截取有关参数，这两个值需要计算
jielen=1200;
%% 读数据
file=dir(fullfile(str1,'*.mat'));
[n,~]=size(file);
%% 按顺序裁剪图象
for i=1:n
    i                                    %输出便于观察顺序
    mdata0=load([str1,file(i).name]);    %读数据
    strr=file(i).name(1:end-4);          %代表存储文件地址名字
    data=[mdata0.cpp];
    data=(data-mmin)*1/(mmax-mmin);       %归一化
    [ss,len]=size(data);
    %%
    fchx=zeros(1,ss);                   %初始化一个一维数组
    fchhx=zeros(1,ss-jies+1);
    for j=1:ss
        fchx(j)=var(data(j,:));         %求每一行方差
    end
    quan=(1-(jies/2-1)*0.001):0.001:1;
    qzh=[quan,fliplr(quan)];             %镜像翻转组合
    for j=1:ss-jies+1
        fchhx(j)=sum(sum(fchx(j:j+jies-1).*qzh));
    end
    [x,yx]=max(fchhy);                  %找出方差最大位置，行数
    %%
    fchy=zeros(1,len);
    fchhy=zeros(1,len-jielen+1);
    for j=1:len
        fchy(j)=var(data(:,j));
    end
    quan=(1-(jielen/2-1)*0.001):0.001:1;
    qzh=[quan,fliplr(quan)];
    for j=1:len-jielen+1
        fchhy(j)=sum(sum(fchy(j:j+jielen-1).*qzh));
    end
    [x,yy]=max(fchhy);               %找列的位置
    %%
    c=data((yx):(yx+jies-1),(yy):(yy+jielen-1));
    cqq=imresize(c,[ss,ss],'bicubic');
    save(strcat(str2,strr),'cqq');
    imwrite(cqq,strcat(str2,strr,'.png'));
    delete([str1,file(i).name]);
end