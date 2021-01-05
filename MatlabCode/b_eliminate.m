clear;
clc;
close all;
fclose all;
format long e;
%% ��������������ļ�
str0='F:\wr20201211\208894Ԥ����\';
str1=strcat(str0,'duiqi\');
str2=strcat(str0,'back\');
load max_15;                   %����a_adjust�д���õ��ļ�ֵ
mmax=max15(1);
mmin=max15(2);
bamax=-2;                      %�ȶ��屳����ʼ��ֵ
bamin=1;
num=100;
scale=6;
%% ��ȡ����
file=dir(fullfile(str1,'*.mat'));
[n,~]=size(file);
%%
for i=1:n
    i                                      %������ڹ۲����
    mdata0=load([str1,file(i).name]);      %��ȡ��ά����ss*length������ͼ����Ϣ
    strr=file(i).name(1:end-4);            %�ļ��洢��
    data=[mdata0.datap];                   %����û����.datap����һ��Ŀ��Ϊ�Σ��������ݸ�ʽ����
    data=(data-mmin)*1000/(mmax-mmin);     %Ϊ��Ҫ����1000
    [ss,len]=size(data);
    fch=zeros(ss,1);
    for j=1:ss
        fch(j)=var(data(j,:));             %��ÿһ�з���
    end
    [x,y]=max(fch);                        %����ϴ�ֵ����Ϊϸ������λ�ã�yΪϸ����������
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
        cdata=data(xst:xend,xxx(j):yyy(j));        %%������xst-xendĬ����ϸ��
        [ss_c,len_c]=size(cdata);
        len_cmid=round(len_c/2);
        cdata_1=(sum(cdata(:,len_cmid-4:len_cmid+5)')/10)';    %���м�10��ƽ��ֵ
        cdata=cdata-repmat(cdata_1,[1,len_c]);
        cxx(j)=std2(cdata);                          %���׼��
    end
    [x,y]=min(cxx);                                  %���׼����С����
    yst=xxx(y);
    yend=yyy(y);
    %% ȥ����
    dataxy=data(xst:xend,yst:yend);           %��ȡ��data��һ��������
    [ss,len]=size(dataxy);
    xxx=[1,round(len*1/5)+1,round(len*2/5)+1,round(len*3/5)+1,round(len*4/5)+1];      %��dataxy��Ϊ5������
    yyy=[round(len*1/5),round(len*2/5),round(len*3/5),round(len*4/5),len];
    cxx=zeros(5,1);
    for j=1:5
        cdata=dataxy(:,xxx(j):yyy(j));
        [ss_c,len_c]=size(cdata);
        len_cmid=round(len_c/2);
        cdata_1=(sum(cdata(:,len_cmid-4:len_cmid+5)')/10)';    %���м�10��ƽ��ֵ
        cdata=cdata-repmat(cdata_1,[1,len_c]);
        cxx(j)=std2(cdata);                          %���׼��
    end
    [x,y]=min(cxx);                                  %��ȡ��������С������λ��
    yyst=xxx(y)+yst-1;                               %�ҵ���������λ��
    yyend=yyy(y)+yst-1;                              %yyend-yyst=1/25���ݳ��ȣ���1/25ͼ�󳤶�
    yrp=round((yyst+yyend)/2);
    [ss,len]=size(data);
    cst=(sum(data(:,yrp-17:yrp+18)')/36)';           %����
    cpp=data-repmat(cst,[1,len]);                    %ȥ����
    
    %% ���¼�ֵ����
    maxp=max(max(cpp));
    minp=min(min(cpp));
    if bamax<maxp
        bamax=maxp;
    end
    if bamin>minp
        bamin=minp;
    end
    %% �����ݺ�ͼ��
    save(strcat(str2,strr),'cpp');
    c=cpp;
    [~,lenp]=size(c);             %������c��ss*lenp�ľ���
    c=(c-minp)*1/(maxp-minp);
    ph=imresize(c,[ss,round(lenp/scale)],'bicubic');       %���µ��������С
    imwrite(ph,strcat(str2,'img',strr,'.png'));            %��ͼƬ
    maxback=[bamax,bamin];                                 %���¼�ֵ
    save('max_back','maxback');
    delete([str1,file(i).name]);                           %ɾ��ԭʼ����
end