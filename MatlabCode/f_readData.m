function [data,row,col] = f_readData(filename)
% F_READDATA read data from file 
%   [DATA,ROW,COL] = F_READDATA(FILENAME) read data from file, whose format
%   can be .Wfm.bin | .csv etc. It also returns two dimension size of the
%   data named ROW and COL.
%
    if(strfind(filename,'.bin')>0)
        file = fopen(filename,'rb');
        [data,num] = fread(file,'float32');
        
        % remove the header information of the bin data
%         data = data(30:end,:);  
        
        [row,col] = size(data);
        pause(0.001)
        fclose(file);
    end
    if(strfind(filename,'.csv')>0)
        data = readmatrix(filename);
        [row,col] = size(data);
    end
end