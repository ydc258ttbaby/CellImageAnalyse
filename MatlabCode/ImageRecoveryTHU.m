function Image = ImageRecoveryTHU(RawData)
    format long e;
    ss=500;                                   %��һ�������ڵ����й�
    delta=[-2:1:2];                           %ssֵ���ܲ��̶���������-2~+2���

    mdata=RawData;                    %�ڶ��д����ѹ
    maxp=max(mdata);                      %��ȡ��ֵ
    minp=min(mdata);
    th=0.16*(maxp-minp)+minp;             %�趨��ֵ��ֵ��
    data=double(mdata>th);
    %% ��������������
    datast=[mdata(1:ss);mdata(1:ss);mdata(1:ss)];   %������չ��һ���������ݣ������ҵ���������
    cha1=zeros(ss,1);
    for j=1:ss
        cha1(j)=sum(datast(ss+j-12:ss+j+12));     %������24�����ݵĺͣ������ҵ���͵�
    end
    [x,y]=min(cha1);                             %����Сֵλ��
    mdata=mdata(y:end);                          %���ݿ�ʼ��֤���°��ߣ����μ����м�
    data=data(y:end);
    %% ���ζ���
    data_len=length(data);
    len=round(data_len/ss);                      %�������ڳ���
    dtdq=data(1:ss);                             %��һ�����ڲ���
    zhizhen=ss+1;
    Image=zeros(ss,len+30);                      %��ʼ��һ����ά����
    Image(:,1)=mdata(1:ss);
    for j=2:len+15
        cha2=zeros(5,1);
        for jj=1:5
            zhizhenj=zhizhen+delta(jj);
            dataj=data(zhizhenj:zhizhenj+ss-1);
            cha2(jj)=sum(sum((dataj-dtdq).^2));   %������С���˷�����
        end
        [x,y]=min(cha2);                           %�����������ݣ���ɶ�ά����
        zhizhen=zhizhen+delta(y);
        Image(:,j)=mdata(zhizhen:zhizhen+ss-1);
        zhizhen=zhizhen+ss;
        if (zhizhen+ss-1+2)>data_len
            break
        end
    end
    if j<(len+30)
        Image(:,(j+1):end)=[];
    end
    Background1 = mean(Image(:, 50 : 250), 2);
    Background2 = mean(Image(:, end - 250 : end-50), 2);
    if mean(Background1) >= mean(Background2)
        Background = Background1;
    else
        Background = Background2;
    end
    
    Image = Image - repmat(Background, 1, size(Image,2));
end

    
    