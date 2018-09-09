clc;
clear;
close all;
%%
% set running flags
image_display_flag  = true;
store_matrix_flag   = true;
permute_on_flag     = false;
maxepochs           = 1;
verbose             = 2;
tolcost             = 1e-8;

% set dataset
load fmri.mat
%%

% load tensor (and equivalent matrix)
Tensor_Y_Noiseless = sol_yxzt(50:240,40:220,:,:);
tensor_dims = size(Tensor_Y_Noiseless);

%%
% set paramters
forget_factor_increments = 4;
sliding_increments = 3;
rank_increments = 3;
observed_increments = 3;
sub_infos_brst = cell(forget_factor_increments,sliding_increments,rank_increments,observed_increments);
parfor forget_factor_plus = 1:forget_factor_increments
    for sliding_size = 1:sliding_increments
        for rank_plus = 1:rank_increments
            for observed_entries_plus = 1:observed_increments
                forgetting_factor = .59+.1*forget_factor_plus;
                sliding_window_size = 2+2*sliding_size;
                max_rank = 20+10*rank_plus;
                fraction = .2+.2*observed_entries_plus;
                %% Build sample tensor
                OmegaTensor = zeros(size(Tensor_Y_Noiseless));
                temp = randsample(prod(tensor_dims),floor(fraction*prod(tensor_dims)));
                OmegaTensor(temp) = 1;

                %% Run test
                loop_string = ['Forgetting factor ',num2str(forgetting_factor),'Sliding window size ',num2str(sliding_window_size),'Rank ',num2str(max_rank),'Observed entries ',num2str(fraction)];
                sub_infos_brst{forget_factor_plus,sliding_size,rank_plus,observed_entries_plus} ={fmri_brst_wrapper(Tensor_Y_Noiseless,fraction,forgetting_factor,sliding_window_size,max_rank),loop_string};
          %      sub_infos_brst{2,forget_factor_plus,sliding_size,rank_plus,observed_entries_plus} = loop_string;
                disp(loop_string)
            end
        end
    end
end
%%
temp = sub_infos_brst{1,1,1,1};
test_temp = temp{1};
figure
title(temp{2})
hold on
for i = 1:z_slices
    for j = 1:time_slices
        subplot(z_slices,time_slices,1);
        imagesc(test_temp.LplusS(:,:,i,j))

    end
end
        %temp1 = sub_infos_brst{1,1,1,1};

%save('fMRI_Completion_sample.15_rank15_window_20')

%% Display images
close all
i = 1;
j= 10;
figure
hold on 
subplot(1,2,1)
imagesc(Tensor_Y_Noiseless(:,:,i,j))
subplot(1,2,2)
imagesc(sub_infos_brst.LplusS(:,:,i,j))

hold off


%%
fraction = 'help'
observe = 100 * (1 - fraction);
if image_display_flag
    figure;
    width = 5;
    height = 3;
    for i=1:total_slices
        
        display_images(rows, cols, observe, height, width, 1, i, sub_infos_TeCPSGD, 'TeCPSGD',0);
        display_images(rows, cols, observe, height, width, 3, i, sub_infos_olstec, 'OLSTEC',0);
        display_images(rows, cols, observe, height, width, 5, i, sub_infos_brst, 'Proposed',1);
        pause(0.1);
    end
end


function display_images(rows, cols, observe, height, width, test, frame, sub_infos, algorithm,proposed_flag)

if(proposed_flag)
   
    
    subplot(height, width, 1 + (test-1));
    imagesc(sub_infos.I(:,:,frame));
    colormap(gray);axis image;axis off;
    title([algorithm, ': ', num2str(observe), '% missing']);
    
    subplot(height, width, width + 1 + (test-1));
    imagesc(sub_infos.L(:,:,frame));
    colormap(gray);axis image;axis off;
    title(['Low-rank image: f = ', num2str(frame)]);
    
    
    
else
    
    
    subplot(height, width, 1 + (test-1));
    imagesc(reshape(sub_infos.I(:,frame),[rows cols]));
    colormap(gray);axis image;axis off;
    title([algorithm, ': ', num2str(observe), '% missing']);
    
    subplot(height, width, width + 1 + (test-1));
    imagesc(reshape(sub_infos.L(:,frame),[rows cols]));
    colormap(gray);axis image;axis off;
    title(['Low-rank image: f = ', num2str(frame)]);
    
    subplot(height, width, 2*width + 1 + (test-1));
    imagesc(reshape(sub_infos.E(:,frame),[rows cols]));
    colormap(gray);axis image;axis off;
    title(['Residual image: error = ', num2str(sub_infos.err_residual(frame))]);
    
    
end


end












