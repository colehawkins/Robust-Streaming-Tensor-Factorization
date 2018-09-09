
% set running flags
image_display_flag  = true;
store_matrix_flag   = true;
permute_on_flag     = false;
maxepochs           = 1;
verbose             = 2;
tolcost             = 1e-8;

% set paramters
max_rank                = 10;
fraction            = .5;

% set dataset

load SAND_TM_Estimation_Data


% load tensor (and equivalent matrix)
Tensor_Y_Noiseless = reshape(X',11,11,2016);
cutoff_size = 100;
Tensor_Y_Noiseless = Tensor_Y_Noiseless(:,:,1:cutoff_size);

tensor_dims = size(Tensor_Y_Noiseless);

% Build sample tensor
    OmegaTensor = zeros(size(Tensor_Y_Noiseless));
temp = randsample(prod(tensor_dims),floor(fraction*prod(tensor_dims)));
OmegaTensor(temp) = 1;



    
    rows = tensor_dims(1);
    cols = tensor_dims(2);
    total_slices = tensor_dims(3);

% revise tensor_dims
tensor_dims(1) = rows;
tensor_dims(2) = cols;
tensor_dims(3) = total_slices;

% set paramter for matrix case
numr = tensor_dims(1) * tensor_dims(2);
numc = tensor_dims(3);

%%

% generate init data
Xinit.A = randn(tensor_dims(1), max_rank);
Xinit.B = randn(tensor_dims(2), max_rank);
Xinit.C = randn(tensor_dims(3), max_rank);

% TeCPSGD
clear options;
options.maxepochs       = maxepochs;
options.tolcost         = tolcost;
options.lambda          = 0.001;
options.stepsize        = 0.1;
options.mu              = 0.05;
options.permute_on      = permute_on_flag;
options.store_subinfo   = true;
options.store_matrix    = store_matrix_flag;
options.verbose         = verbose;

tic;
[Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, max_rank, Xinit, options);
elapsed_time_tecpsgd = toc;


% OLSTEC
clear options;
options.maxepochs       = maxepochs;
options.tolcost         = tolcost;
options.permute_on      = permute_on_flag;
options.lambda          = 0.7;  % Forgetting paramter
options.mu              = 0.1;  % Regualization paramter
options.tw_flag         = 1;    % 0:Exponential Window, 1:Truncated Window (TW)
options.tw_len          = 10;   % Window length for Truncated Window (TW) algorithm
options.store_subinfo   = true;
options.store_matrix    = store_matrix_flag;
options.verbose         = verbose;

tic;
[Xsol_olstec, infos_olstec, sub_infos_olstec] = olstec(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, max_rank, Xinit, options);
elapsed_time_olstec = toc;
%%
% BRST

forgetting_factor = .98;
sliding_window_size = 20;

sub_infos_brst = brst_wrapper(Tensor_Y_Noiseless,fraction,forgetting_factor,sliding_window_size,max_rank);
disp('job done')
%%
%save('Network_Traffic_Completion')




%% plotting

burn_in = 12;
figure;
hold on;
plot(sub_infos_TeCPSGD.inner_iter(burn_in:end), sub_infos_TeCPSGD.err_residual(burn_in:end), '-b', 'linewidth', 2.0);
plot(sub_infos_olstec.inner_iter(burn_in:end), sub_infos_olstec.err_residual(burn_in:end), '-r', 'linewidth', 2.0);
plot(sub_infos_brst.inner_iter(burn_in:end), sub_infos_brst.err_residual(burn_in:end), '-G', 'linewidth', 2.0);
hold off;
grid on;
legend('Online-SGD', 'OLSTEC','Proposed');
ax1 = gca;
set(ax1,'FontSize',20);
xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
ylabel('normalized residual error','FontName','Arial','FontSize',fs,'FontWeight','bold');













