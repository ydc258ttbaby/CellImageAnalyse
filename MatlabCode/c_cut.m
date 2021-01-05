clear;
clc;
close all;
fclose all;
format long e;
%% ��������������ļ�
str0='F:\wr20201211\208894Ԥ����\';
str1=strcat(str0,'back\');       %ԭʼ��������λ��
str2=strcat(str0,'cut\');        %��ͼ����λ��
load max_back;                   %��b_eliminate�д���õ��ļ�ֵ
mmax=max15(1);
mmin=max15(2);
jies=100;                        %�ͽ�ȡ�йز�����������ֵ��Ҫ����
jielen=1200;
%% ������
file=dir(fullfile(str1,'*.mat'));
[n,~]=size(file);
%% ��˳��ü�ͼ��
for i=1:n
    i                                    %������ڹ۲�˳��
    mdata0=load([str1,file(i).name]);    %������
    strr=file(i).name(1:end-4);          %����洢�ļ���ַ����
    data=[mdata0.cpp];
    data=(data-mmin)*1/(mmax-mmin);       %��һ��
    [ss,len]=size(data);
    %%
    fchx=zeros(1,ss);                   %��ʼ��һ��һά����
    fchhx=zeros(1,ss-jies+1);
    for j=1:ss
        fchx(j)=var(data(j,:));         %��ÿһ�з���
    end
    quan=(1-(jies/2-1)*0.001):0.001:1;
    qzh=[quan,fliplr(quan)];             %����ת���
    for j=1:ss-jies+1
        fchhx(j)=sum(sum(fchx(j:j+jies-1).*qzh));
    end
    [x,yx]=max(fchhy);                  %�ҳ��������λ�ã�����
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
    [x,yy]=max(fchhy);               %���е�λ��
    %%
    c=data((yx):(yx+jies-1),(yy):(yy+jielen-1));
    cqq=imresize(c,[ss,ss],'bicubic');
    save(strcat(str2,strr),'cqq');
    imwrite(cqq,strcat(str2,strr,'.png'));
    delete([str1,file(i).name]);
end