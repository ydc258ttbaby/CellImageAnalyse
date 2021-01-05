clear;
clc;
close all;
fclose all;
format long e
file=dir(fullfile('F:\0808duiqi\*.mat'));
[n,~]=size(file);
str1='F:\0808duiqi\';
    
for i=1:n
    i
    mdata0=load([str1,file(i).name]);
    str2=file(i).name(1:end-4);
    data=[mdata0.datap];
    [ss,len]=size(data);
    fch=zeros(ss,1);
    for j=1:ss
        fch(j)=var(data(j,:));
    end
    [x,y]=max(fch);
    xst=y-5;
    xend=y+5;
    if xst<1
        xst=1;
    end
    if xend>len
        xend=len;
    end
    
    cxx=zeros(8,1);
    xxx=[1,round(len*1/8),round(len*1/4),round(len*3/8),round(len*4/8),round(len*5/8),round(len*3/4),len-159];
    yyy=xxx+79;
    for j=1:8
        cdata=sum(data(xst:xend,xxx(j):xxx(j)+159))/11;
        cxx(j)=var(cdata);
    end
    [x,y]=min(cxx);
    yrp=yyy(y);
    
    
    cst=(sum(data(:,yrp-4:yrp+5)')/10)';
    cqq=data-repmat(cst,[1,len]);
    save(strcat('F:\0808backg\',str2),'cqq');

end
fclose all;
load chirp;
% sound(y,Fs);
imagesc(cqq');colormap(gray);