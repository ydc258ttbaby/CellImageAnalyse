close all
clear all
filePath = "D:\天津\图像数据\229_33k\160\"
desPath = strcat(filePath,'sampleImg');
mkdir(desPath);
srcnamelist = dir(strcat(filePath,'*.png'));
L = length(srcnamelist);
randomIndex = round(rand(1,L)*L);
randomIndex = unique(randomIndex);
count = 1;
for i = 1:length(randomIndex)
   if(count > 5000)
       break;
   end
   count = count + 1
   if(randomIndex(i)>0 && randomIndex(i)<=L)
       index = randomIndex(i);
       movefile(strcat(filePath,srcnamelist(index).name),strcat(desPath,'\',srcnamelist(index).name),'f');
   end
end
