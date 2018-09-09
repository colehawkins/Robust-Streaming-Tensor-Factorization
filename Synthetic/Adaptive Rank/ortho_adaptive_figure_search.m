close all
clear all
clc

rng('default');
true_rank = 5;

max_rank = true_rank;

sz = [50,50,50];

stream_length = 30;
%drift_percent = .01;
error_percentage = .02;


stream = ortho_stream_generator(true_rank,sz,stream_length);%,drift_percent);

temp = stream{1,1};
error_size = 10*mean(temp(:)); 
error_var = 0;

noise_mean = 0;
noise_var = 10^(-3);

error_stream = error_stream_generator(sz,stream_length,error_percentage,error_size,error_var);
noise_stream = noise_stream_generator(sz,stream_length,noise_mean,noise_var);
stream_with_errors = cell(1,stream_length);

for i = 1:stream_length
    stream_with_errors{i} = stream{1,i}+error_stream{i}+noise_stream{i};
end

%%
observed_ratios = [.15,.5,.8];
observed_tests = length(observed_ratios);

window_sizes = [20];
window_tests = length(window_sizes);

forget_factors = [.9,.99];
forget_tests = length(forget_factors);

proposed_factorization = cell(observed_tests,window_tests,forget_tests);

ml_init = 1;

for kk = 1:observed_tests
    
    sample_stream = sample_stream_generator(sz,stream_length,observed_ratios(kk)); 
    
    for ii = 1:window_tests
        
        sliding_window_size = window_sizes(ii);
        
        parfor jj = 1:forget_tests
            
            forgetting_factor = forget_factors(jj);
            
            proposed_factorization{kk,ii,jj} = streaming_bayesian_completion(stream_with_errors,sample_stream,forgetting_factor,sliding_window_size,ml_init,max_rank);
             disp([jj,ii,kk])
        end
    end
end
disp('job done')
%%
for kk = 1:observed_tests
    
   % sample_stream = sample_stream_generator(sz,stream_length,observed_ratios(kk)); 
    
    for ii = 1:window_tests
        
    %    sliding_window_size = window_sizes(ii);
        
        for jj = 1:forget_tests
            
     %       forgetting_factor = forget_factors(jj);
            figure
            adaptive_rank_figure_generator(proposed_factorization{kk,ii,jj},observed_ratios(kk),window_sizes(ii),forget_factors(jj));
           
        end
    end
end


%%
% for i = 1:stream_length
%     temp1 = stream{1,i};
%     temp2 = ktensor(proposed_factorization{1,i});
%     multi_test_relative_error(1,1,i) = norm(temp1(:)-temp2(:))/norm(temp1(:));
% end
% 
% 
% %save('synthetic_completion_tests')
% disp('job done')
% adaptive_rank_figure_generator(proposed_factorization)

% %% plot results
% close all
% ts = [1:TT]; %for online CP
% 
% 
% % error vs steps
% figure
% hold on;
% plot(num_init_slices+ts,relative_error(1,(num_init_slices+1):end),'-k','LineWidth',2);
% 
% plot(num_init_slices+ts,relative_error(2,(num_init_slices+1):end),'-s','LineWidth',2);
% xlabel('Steps');
% ylabel('Relative Error');
% legend('OnlineCP','Proposed');
% grid on;
% set(gca,'FontSize',14)
% adaptive_rank_figure_generator(proposed_factorization)