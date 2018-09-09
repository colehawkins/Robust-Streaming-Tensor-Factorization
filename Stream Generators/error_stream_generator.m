function error_stream = error_stream_generator(sz,frames,percentage,magnitude,variance)

error_stream = cell(1,frames);

for j = 1:frames
    temp = zeros(sz);
    error_location = datasample(1:prod(sz),floor(percentage*prod(sz)));
    
    for i = 1:length(error_location)
       temp(error_location(i)) = normrnd(magnitude,variance); 
    end
    error_stream{j} = temp;
    
end

end