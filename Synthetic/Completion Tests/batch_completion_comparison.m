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

%% Test batch

%build big tensor

full_tensor = zeros([sz,stream_length]);
full_sample_tensor = ones([sz,stream_length]);
ground_truth = zeros([sz,stream_length]);
for i = 1:stream_length
    full_sample_tensor(:,:,i) = sample_stream{i};
    full_tensor(:,:,i) = stream_with_errors{i};
    ground_truth(:,:,i) = stream{1,i};
end

batch_out = BayesRCP_TC(full_tensor,'obs',full_sample_tensor,'maxRank',max_rank);

batch_error = zeros(1,stream_length);

for i = 1:stream_length
    temp = batch_out.Z;
    temp1 = {temp{1},temp{2},temp{3}(i,:)};
    built = kr(temp1);
    temp2 = stream{1,i};
    batch_error(i) =  norm(built(:)-temp2(:))/norm(temp2(:));
end


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
save('batch_completion_comparison_workspace');


%% plot results
close all


%% plot comparison against recieved tensor with errors

figure
hold on;

plot(1:stream_length,batch_error(1,:),'-s','LineWidth',2);

plot(1:stream_length,proposed_relative_error(2,:),'-s','LineWidth',2);


xlabel('Steps','fontweight','bold','fontsize',16);
ylabel('Relative Error','fontweight','bold','fontsize',16);
%title('Relative Error on Ground Truth','fontweight','bold','fontsize',16);
legend('Batch','Proposed');
grid on;
set(gca,'FontSize',14)
