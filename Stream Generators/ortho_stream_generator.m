function stream = ortho_stream_generator(rank,sz,frames)%,drift_percent)
%stream = stream_generator(rank,sz,frames,drift)
%stream{1,:} = the output tensor
%stream{2,:} = the factors used

start_factors = ortho_cp_factors(rank,sz);
end_factors= ortho_cp_factors(rank,sz);
%ortho_cp_factors(rank,sz);

h = 1/(4*frames);

stream = cell(2,frames);


for j = 1:frames
    
    t = j*h;
    temp = {(1-t)*start_factors{1}+(t)*end_factors{1}};
    
    for i = 2:length(sz)    
        
        temp = [temp,(1-t)*start_factors{i}+(t^2)*end_factors{i}];
        stream{2,j} = temp;
        
    end
    stream(1,j)={build_cp_tensor(stream{2,j})};
    
end


end