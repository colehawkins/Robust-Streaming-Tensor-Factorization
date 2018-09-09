close all
clear all
clc

rng('default');
true_rank = 5;

max_rank = true_rank;

sz = [40,40];

stream_length = 100;
%drift_percent = .01;
error_percentage = .02;

stream = stream_generator_updated(true_rank,sz,stream_length);%,drift_percent);

temp = stream{1,1};
error_size = 10*mean(temp(:)); 
error_var = 0;

noise_mean = 0;
noise_var = 10^(-2);

error_stream = error_stream_generator(sz,stream_length,error_percentage,error_size,error_var);
noise_stream = noise_stream_generator(sz,stream_length,noise_mean,noise_var);
stream_with_errors = cell(1,stream_length);

for i = 1:stream_length
    stream_with_errors{i} = stream{1,i}+error_stream{i}+noise_stream{i};
end

num_init_slices = 5;

observed = .15;

sample_stream = sample_stream_generator(sz,stream_length,observed);


%% Test OLSTEC

%build big tensor

full_tensor = zeros([sz,stream_length]);
full_sample_tensor = ones([sz,stream_length]);
ground_truth = zeros([sz,stream_length]);
for i = 1:stream_length
    full_sample_tensor(:,:,i) = sample_stream{i};
    full_tensor(:,:,i) = stream_with_errors{i};
    ground_truth(:,:,i) = stream{1,i};
end

[Xsol, infos, sub_infos] = olstec_altered(full_tensor,full_sample_tensor,[],[sz,stream_length],true_rank,[],ground_truth,[]);

OLSTEC_relative_error = zeros(2,stream_length);
OLSTEC_relative_error(1,:) = sub_infos.err_residual(2:end);
OLSTEC_relative_error(2,:) = sub_infos.err_residual_truth(2:end);

%% Test Online-SGD
clear options;
maxepochs           = 1;
verbose             = 2;
tolcost             = 1e-8;
image_display_flag  = true;
store_matrix_flag   = true;
permute_on_flag     = false;
options.maxepochs       = maxepochs;
options.tolcost         = tolcost;
options.lambda          = 0.001;
options.stepsize        = 0.1;
options.mu              = 0.05;
options.permute_on      = permute_on_flag;
options.store_subinfo   = true;
options.store_matrix    = store_matrix_flag;
options.verbose         = verbose;

Xinit.A = randn(sz(1), max_rank);
Xinit.B = randn(sz(2), max_rank);
Xinit.C = randn(stream_length, max_rank);

[SGDsol, SGDinfos, SGDsub_infos] = TeCPSGD_altered(full_tensor,full_sample_tensor,[],[sz,stream_length],true_rank,Xinit,ground_truth, options);

OSGD_relative_error = zeros(2,stream_length);

OSGD_relative_error(1,:) = SGDsub_infos.err_residual(2:end);
OSGD_relative_error(2,:) = SGDsub_infos.err_residual_truth(2:end);

%% Test Proposed

forgetting_factor = .8;
sliding_window_size = 10;
ml_init = 1;
sample_stream = cell(1,stream_length);

for i = 1:stream_length
    sample_stream{i} = ones(sz);
end

proposed_factorization = streaming_bayesian_completion(stream_with_errors,sample_stream,forgetting_factor,sliding_window_size,ml_init,true_rank);

proposed_relative_error = zeros(2,stream_length);

for i = 1:stream_length
    temp1 = stream_with_errors{1,i};
    temp2 = double(ktensor(proposed_factorization{1,i}));
    proposed_relative_error(1,i) = norm(temp1(:)-temp2(:))/norm(temp1(:));
    temp1 = stream{1,i};
    proposed_relative_error(2,i) = norm(temp1(:)-temp2(:))/norm(temp1(:));
end
%%
save('synthetic_completion_comparison_workspace')


%% plot results
close all


%% plot comparison against recieved tensor with errors

figure
hold on;

plot(1:stream_length,OLSTEC_relative_error(2,:),'-s','LineWidth',2);

plot(1:stream_length,OSGD_relative_error(2,:),'-s','LineWidth',2);

plot(1:stream_length,proposed_relative_error(2,:),'-s','LineWidth',2);


xlabel('Steps','fontweight','bold','fontsize',16);
ylabel('Relative Error','fontweight','bold','fontsize',16);
legend('Olstec','Online-SGD','Proposed');
grid on;
set(gca,'FontSize',14)
