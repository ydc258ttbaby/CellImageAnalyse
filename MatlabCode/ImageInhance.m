% I = imread('C:\ydc\学习\实验室\天津院\图像数据\580\580__262144_2438.png');
close all
I=imread('C:\ydc\学习\实验室\天津院\图像数据\580\580__262144_1097.png');
J=histeq(I);  %直方图均衡化，这一个函数就可以做到均衡化的效果
figure,
subplot(121),imshow(uint8(I));
title('原图')
subplot(122),imshow(uint8(J));
title('均衡化后')
figure,
subplot(121),imhist(I,64);
title('原图像直方图');
subplot(122),imhist(J,64);
title('均衡化后的直方图');
