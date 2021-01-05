close all;
[Image, RepRate] = ImageRecoveryModify(data, 20000000,1,1.0,0.7);
h2 = figure;
set(h2,'position',[100 100 1200 720]);
% h1 = subplot(121)
% plot_1D_Single(data,'data');
% h2 = subplot(122)
imagesc(Image);
colormap(gray);
% line([200,200+10/0.06],[50;50],'linewidth',2,'color','r')
% set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w')
% text(200,65,'10um','color','r')
% print('-dpng',strcat('C:\ydc\学习\实验室\天津院\图像数据\523\',imgName,num2str(j),'.png'));
% pause(0.01)
j=j+1;

