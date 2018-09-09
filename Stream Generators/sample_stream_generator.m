function stream = sample_stream_generator(sz,frames,ratio)
%stream = sample_stream_generator(sz,frames,ratio)
%sz = the size of a single tensor slice
%frames = number of frames
%ratio = observed_ratio

stream = cell(1,frames);

amount = ceil(ratio*prod(sz));

for j = 1:frames
    
    entries = datasample(1:prod(sz),amount,'Replace',false);
    temp = zeros(sz);
    temp(entries) = 1;
    stream(j)={temp};
end

end