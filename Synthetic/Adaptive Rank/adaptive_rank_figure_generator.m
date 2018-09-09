function adaptive_rank_figure_generator(proposed_factorization,observed_ratio,window_size,forget_factor)

    num_slices = length(proposed_factorization(1,:));
    rank_estimates = zeros(1,num_slices);
    for i = 1:num_slices
        %temp = proposed_factorization;
       rank_estimates(i) = rank_estimate(proposed_factorization{1,i}{1});
    end
    
    plot(rank_estimates)
    xlabel('Slice')
    ylabel('Rank')
    title(['Observed ratio ',num2str(observed_ratio),' Window size ',num2str(window_size),' Forget factor ',num2str(forget_factor)]);
    

    function estimate = rank_estimate(in_mat)
       estimate = 0;
       for ii = 1:length(in_mat(1,:))
          if norm(in_mat(:,ii))>eps
            estimate = estimate+1;
          end
          
       end
    end
end
          