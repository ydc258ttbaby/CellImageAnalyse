clc;
clear all;
close all;
srcPath = 'C:\ydc\学习\实验室\天津院\图像数据\';
srcnamelist = dir(strcat(srcPath,'*.fig'));
for i = 1:length(srcnamelist)
   close all;
   open(strcat(srcPath, srcnamelist(i).name));
   print('-dpng',strcat('p_',num2str(i),'.png'));
end