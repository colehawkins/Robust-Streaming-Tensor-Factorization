function stream = stream_generator(rank,sz,frames,drift_percent)
%stream = stream_generator(rank,sz,frames,drift)
%stream{1,:} = the output tensor
%stream{2,:} = the factors used

factors = ortho_cp_factors(rank,sz);%rand_cp_factors(rank,sz);
stream = cell(2,frames);


perturbed_factors = perturb(factors);
stream(1,1)={build_cp_tensor(perturbed_factors)};
stream(2,1)={perturbed_factors};



for j = 2:frames
    perturbed_factors = perturb(stream{2,j-1});
    stream(1,j)={build_cp_tensor(perturbed_factors)};
    stream(2,j)={perturbed_factors};
end


    function out = perturb(factors) 
        order = length(factors);
        out = cell(1,order);
        
        for i = 1:order
            factor_norm = norm(factors{i});
            perturbation = rand(size(factors{i}))-1/2;
            perturbation = perturbation/norm(perturbation(:));
            perturbation = factor_norm*drift_percent*perturbation;
            out{i} = factors{i} +perturbation;
        end
    end
end