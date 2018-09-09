clc; clear; close all;


% Set the video information
videoSequence = 'BUS_176x144_15_orig_01 (1).yuv';
width  = 176;
height = 144;
nFrame = 15;
%

% Read the video sequence
[Y,U,V] = yuvRead(videoSequence, width, height ,nFrame); 

%%
temp = uint8(zeros(144,176,3,15));

%%
for i = 1:15
    temp(:,:,:,i) = yuv2rgb(Y(:,:,i),U(:,:,i),V(:,:,i));
end

to_complete = double(temp);
%%

fraction            = 1;

% set dataset
data_type   = 'static';%'static'; % 'dynamic';

% set paramters
max_rank                = 15;


%%
% BRST

forgetting_factor = .98;
sliding_window_size = 20;

sub_infos_brst = brst_wrapper(to_complete,fraction,forgetting_factor,sliding_window_size,max_rank);
disp('job done')

%%
figure;
hold on;
semilogy(sub_infos_TeCPSGD.inner_iter, sub_infos_TeCPSGD.err_run_ave, '-b', 'linewidth', 2.0);
semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_run_ave, '-r', 'linewidth', 2.0);
semilogy(sub_infos_brst.inner_iter, sub_infos_brst.err_run_ave, '-g', 'linewidth', 2.0);
hold off;
grid on;
legend('TeCPSGD', 'OLSTEC','Proposed', 'location', 'best');
%legend('TeCPSGD', 'OLSTEC');
ax1 = gca;
set(ax1,'FontSize',fs);
xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
ylabel('running average error','FontName','Arial','FontSize',fs,'FontWeight','bold');

%% Display images
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












