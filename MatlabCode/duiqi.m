% clear;
clc;
close all;
fclose all;
format long e
file=dir(fullfile('*.mat'));
[n,~]=size(file);

mmax=0.05;%%max voltage
mmin=-0.00125;%%min voltage
str1='';
intv=40000;
% for i=1:n
    
%     mdata0=load([str1,file(i).name]);
%     data=[mdata0.mdata];
%     str2=file(i).name(1:end-4);
    data_len=length(data);
    sinyy = data-mean(data);
    yy=data_len/2:intv:data_len;
    count=0;
    for j=data_len/2:intv:data_len
        count=count+1;
        siny =fft(sinyy(401:j));
        ny = length(siny);
        freq=(0:ny)/ny;
        freq=freq(1:floor(ny/2));
        siny=abs(siny(1:floor(ny/2)));
        [~,y]=max(siny);
        prd=1/freq(y);
        yy(count)=prd;
    end
    [maxv,maxl]=findpeaks(yy);
%     prd=mean(yy(end-maxl(end)+maxl(end-1)+1:end));
    prd = mean(yy);
    len=ceil(data_len/prd);
    ss=round(prd);
    prd0=prd;
        datast=data(round(prd0*1)-ss+1:round(prd0*1));
        datamid=data(round(prd0*len*0.75)-ss+1:round(prd0*len*0.75));
        dataend=data(round(prd0*(len-1))-ss+1:round(prd0*(len-1)));
        delt0=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
    for j=-0.0008:0.00004:0.0008
        prdj=prd0+j;
        datast=data(round(prdj*1)-ss+1:round(prdj*1));
        datamid=data(round(prdj*len*0.75)-ss+1:round(prdj*len*0.75));
        dataend=data(round(prdj*(len-1))-ss+1:round(prdj*(len-1)));
        delt=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
        if delt<delt0
            prd=prdj;
            delt0=delt;
        end
    end
    


        len=ceil(data_len/prd);
        ss=round(prd);
        prd0=prd;
        datast=data(round(prd0*1)-ss+1:round(prd0*1));
        datamid=data(round(prd0*len*0.75)-ss+1:round(prd0*len*0.75));
        dataend=data(round(prd0*(len-1))-ss+1:round(prd0*(len-1)));
        delt0=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
    for j=-0.00008:0.000004*0.00008
        prdj=prd0+j;
        datast=data(round(prdj*1)-ss+1:round(prdj*1));
        datamid=data(round(prdj*len*0.75)-ss+1:round(prdj*len*0.75));
        dataend=data(round(prdj*(len-1))-ss+1:round(prdj*(len-1)));
        delt=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
        if delt<delt0
            prd=prdj;
            delt0=delt;
        end
    end
    
    len=ceil(data_len/prd);
        ss=round(prd);
        prd0=prd;
        datast=data(round(prd0*1)-ss+1:round(prd0*1));
        datamid=data(round(prd0*len*0.75)-ss+1:round(prd0*len*0.75));
        dataend=data(round(prd0*(len-1))-ss+1:round(prd0*(len-1)));
        delt0=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
    for j=-0.000008:0.0000004*0.000008
        prdj=prd0+j;
        datast=data(round(prdj*1)-ss+1:round(prdj*1));
        datamid=data(round(prdj*len*0.75)-ss+1:round(prdj*len*0.75));
        dataend=data(round(prdj*(len-1))-ss+1:round(prdj*(len-1)));
        delt=sum(sum(abs(datamid-datast)))+sum(sum(abs(dataend-datast)));
        if delt<delt0
            prd=prdj;
            delt0=delt;
        end
    end
  
    
    len=ceil(data_len/prd);
    ss=round(prd);
    datap=zeros(ss,len);
    for j=1:len
        datap(:,j)=data(round(prd*j)-ss+1:round(prd*j));
        if round(prd*(j+1))>data_len
            break
        end
    end

    if j<len
        datap(:,(j+1):end)=[];
    end
    figure
    imagesc(datap)
    colormap(gray);
%     save(strcat('dq',str2),'datap');

% end
fclose all;
