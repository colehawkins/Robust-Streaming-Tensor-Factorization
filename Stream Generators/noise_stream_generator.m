function error_stream = noise_stream_generator(sz,frames,mu,sigma)

error_stream = cell(1,frames);

for j = 1:frames
    
    error_stream(j) = {normrnd(mu,sigma,sz)};
    
end

end