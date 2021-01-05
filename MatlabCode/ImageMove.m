% srcPath = 'C:\ydc\学习\实验室\天津院\图像数据\578\';
% srcnamelist = dir(strcat(srcPath,'*.png'));
% i = 1;
% isMkdir = 0;
% for j = 1:length(srcnamelist)
%    if isMkdir == 0
%       curtime = datestr(now,'yyyymmdd THHMMSS');
%       curtime(find(isspace(curtime))) = '_';
%       desPath = strcat(srcPath,curtime);
%       mkdir(desPath);   
%       isMkdir = 1;
%    end
%    if i > 1000
%        isMkdir = 0;
%        i=1;
%    end
%    i=i+1;
%    movefile(strcat(srcPath,srcnamelist(j).name),strcat(desPath,'\',srcnamelist(j).name),'f'); 
% end

srcPath = 'C:\ydc\学习\实验室\天津院\图像数据\578\';
srcnamelist = dir(srcPath);
for i = 1:length(srcnamelist)
   sonPath = strcat(srcPath,srcnamelist(i).name,'\');
   sonnamelist = dir(strcat(sonPath,'*.png'));
   for j = 1:length(sonnamelist)
       movefile(strcat(sonPath,sonnamelist(j).name),strcat(srcPath,'\',sonnamelist(j).name),'f'); 
   end
end