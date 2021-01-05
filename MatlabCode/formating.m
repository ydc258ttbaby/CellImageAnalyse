clear;
clc;
close all;
fclose all;
format long e
file=dir(fullfile('F:\0804\b\*.csv'));
[n,~]=size(file);
mmax=-1;
mmin=2;
str1='F:\0804\b\';
for i=1:n
%     i
    mdata0=csvread([str1,file(i).name]);
    strr=file(i).name(1:end-4);
    mdata=mdata0(:,2);
    maxp=max(max(mdata));
    minp=min(min(mdata));
    if mmax<maxp
        mmax=maxp;
    end
    if mmin>minp
        mmin=minp;
    end
    save(strcat('F:\0807\',strr),'mdata');
end

fclose all;
