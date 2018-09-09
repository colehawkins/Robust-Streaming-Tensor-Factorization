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


%% Test OLSTEC

%build big tensor

full_tensor = zeros([sz,stream_length]);
full_sample_tensor = ones([sz,stream_length]);
ground_truth = zeros([sz,stream_length]);
for i = 1:stream_length
   
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

[SGDsol, SGDinfos, SGDsub_infos] = TeCPSGD(full_tensor,full_sample_tensor,[],[sz,stream_length],true_rank,Xinit,options);
OSGD_relative_error = SGDsub_infos.err_residual(2:end);


%% Test Online-CP



N = length(sz)+1;
TT = stream_length-num_init_slices;
R = true_rank;

X = zeros([sz,stream_length]);
for i = 1:stream_length
    X(:,:,i) = stream_with_errors{1,i};
end

idx = repmat({':'}, 1, length(sz)+1);
idx(end) = {1:num_init_slices};
initX = X(idx{:});

% factorize initX
initOpt.printitn = 1;
initOpt.maxiters = 100;
initOpt.tol = 1e-8;
estInitX = cp_als(tensor(initX), R, initOpt);
initAs = estInitX.U;
% absorb lambda into the last dimension
initAs{end} = initAs{end}*diag(estInitX.lambda);


% initialize onlineCP method
[onlinePs, onlineQs] = onlineCP_initial(initX, initAs, R);
onlineAs = initAs(1:end-1);
onlineAs_N = initAs{end};

k = 1;

OnlineCP_relative_error = zeros(2,stream_length-num_init_slices);

for t=1:TT
    clc;
    fprintf('the %dth steps\n', k);
    % get the incoming slice
    endTime = min(num_init_slices+t, stream_length);
    idx(end) = {num_init_slices+t:endTime};
    x = squeeze(X(idx{:}));
    numOfSlice = endTime-num_init_slices-t+1;
    % get tensor X of time current time
    idx(end) = {1:endTime};
    Xt = X(idx{:});
    Xt = stream{1,t};
    % online CP
    [onlineAs, onlinePs, onlineQs, onlineAlpha] = onlineCP_update(x, onlineAs, onlinePs, onlineQs);
    onlineAs_N(end+1,:) = onlineAlpha;
    
    tmp = [onlineAs; {onlineAs_N(end,:)}];
    temp1 = stream_with_errors{1,t};
    temp2 = double(full(ktensor(tmp)));
    OnlineCP_relative_error(1, k) = norm(temp1(:)-temp2(:))/norm(temp1(:));
    temp1 = stream{1,t};
    OnlineCP_relative_error(2, k) = norm(temp1(:)-temp2(:))/norm(temp1(:));
    
    
    k = k+1;
end

%% Test Proposed

forgetting_factor = .99;
sliding_window_size = 7;
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
save('synthetic_factorization_comparison_workspace')


%% plot results
close all


%% plot comparison against recieved tensor with errors

figure
hold on;

ts = [1:TT]; %for online CP
plot(num_init_slices+ts,OnlineCP_relative_error(2,:),'-s','LineWidth',2);

plot(1:stream_length,OLSTEC_relative_error(2,:),'-s','LineWidth',2);

plot(1:stream_length,OSGD_relative_error(1,:),'-s','LineWidth',2);

plot(1:stream_length,proposed_relative_error(2,:),'-s','LineWidth',2);


xlabel('Steps','fontweight','bold','fontsize',16);
ylabel('Relative Error','fontweight','bold','fontsize',16);
%title('Relative Error on Ground Truth','fontweight','bold','fontsize',16);
legend('OnlineCP','Olstec','Online-SGD','Proposed');
grid on;
set(gca,'FontSize',14)

%%

figure
hold on;

ts = [1:TT]; %for online CP
plot(num_init_slices+ts,OnlineCP_relative_error(1,:),'-s','LineWidth',2);

plot(1:stream_length,OLSTEC_relative_error(1,:),'-s','LineWidth',2);

plot(1:stream_length,proposed_relative_error(1,:),'-s','LineWidth',2);


xlabel('Steps','fontweight','bold','fontsize',16);
ylabel('Relative Error','fontweight','bold','fontsize',16);
%title('Relative Error on Corrupted Data','fontweight','bold','fontsize',16);
legend('OnlineCP','Olstec','Proposed');
grid on;
set(gca,'FontSize',14)