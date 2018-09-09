function sub_infos = fmri_brst_wrapper(input_tensor,observed_ratio,forgetting_factor,sliding_window_size,max_rank)

sz = size(input_tensor);
sample_stream = sample_stream_generator(sz(1:(end-1)),sz(end),observed_ratio);
ml_init = 1;

to_complete = cell(1,sz(end));

for i = 1:sz(end)
    to_complete{i} = input_tensor(:,:,:,i);
end

proposed_factorization = streaming_bayesian_completion(to_complete,sample_stream,forgetting_factor,sliding_window_size,ml_init,max_rank);


sub_infos.inner_iter = 0:sz(end);
sub_infos.err_residual = zeros(1,sz(end)+1);
sub_infos.err_run_ave = zeros(1,sz(end)+1);

sub_infos.I = zeros(sz);
sub_infos.L = zeros(sz);



for i = 1:sz(end)
    temp1 = double(ktensor(proposed_factorization{1,i}))+proposed_factorization{2,i}-to_complete{i};
    temp2 = input_tensor(:,:,:,i);
    sub_infos.err_residual(i+1) = norm(temp1(:))/norm(temp2(:));
    sub_infos.err_run_avg(i+1) = mean(sub_infos.err_residual(2:(i+1)));
    
    sub_infos.I(:,:,:,i) = sample_stream{1,i}.*input_tensor(:,:,:,i);
    sub_infos.LplusS(:,:,:,i) = double(ktensor(proposed_factorization{1,i}))+proposed_factorization{2,i};
    sub_infos.L(:,:,:,i) = double(ktensor(proposed_factorization{1,i}));
    
end

    sub_infos.proposed_factorization = proposed_factorization;

end