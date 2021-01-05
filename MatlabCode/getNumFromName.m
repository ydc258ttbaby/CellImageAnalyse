function num = getNumFromName(s)
    index = 22;
    str = '';
    while(1)
        if(s(index)=='_') break;
        end
        str = [str,s(index)];
        s(index);
        index = index + 1;
    end
    num = str2num(str);
end