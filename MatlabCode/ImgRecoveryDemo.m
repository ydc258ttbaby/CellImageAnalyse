close all;
clear all;
clc;
srcPath = 'D:\576\20201015_T144621\';
srcnamelist = dir(strcat(srcPath,'*.Wfm.bin'));
for i = 1:length(srcnamelist)
   filename = strcat( srcPath,srcnamelist(i).name);
   if(strfind(filename,'.Wfm.bin')>0)
        file = fopen(filename,'rb');
        [data,n] = fread(file,'float32');
        pause(0.001)
        fclose(file);
   end
   [Image, RepRate] = ImageRecoveryModify(data, 20000000,1,1.0,0.7);
   close all;
   figure
   imagesc(Image);
   colormap('gray')
   pause(0.5);
   
end