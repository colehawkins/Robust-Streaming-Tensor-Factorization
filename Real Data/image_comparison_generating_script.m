%observe = 100 * (1 - fraction);

i = 10;
close all
figure;
frame1 = 20;
frame2 = 50;

subplot_tight(2, 4, 1,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.I(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Ground Truth');

subplot_tight(2, 4, 2,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.L(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Online-SGD');


subplot_tight(2, 4, 3,[0.01,.01]);
imagesc(reshape(sub_infos_olstec.L(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('OLSTEC');

subplot_tight(2, 4, 4,[0.01,.01]);
imagesc(reshape(sub_infos_brst.L(:,:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Proposed');

subplot_tight(2, 4, 5,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.I(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;


subplot_tight(2, 4, 6,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.L(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;

subplot_tight(2, 4, 7,[0.01,.01]);
imagesc(reshape(sub_infos_olstec.L(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;

subplot_tight(2, 4, 8,[0.01,.01]);
imagesc(reshape(sub_infos_brst.L(:,:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;

%% Completion

i = 10;
close all
figure;
frame1 = 10;
frame2 = 50;

temp = subplot_tight(2, 5, 1,[0.01,.01]);
imagesc(Tensor_Y_Noiseless(:,:,frame1));
colormap(gray);axis image;axis off;
title('Ground Truth');

temp = subplot_tight(2, 5, 2,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.I(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Sampled Entries');

subplot_tight(2, 5, 3,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.L(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Online-SGD');


subplot_tight(2, 5, 4,[0.01,.01]);
imagesc(reshape(sub_infos_olstec.L(:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('OLSTEC');

subplot_tight(2, 5, 5,[0.01,.01]);
imagesc(reshape(sub_infos_brst.L(:,:,frame1),[rows,cols]));
colormap(gray);axis image;axis off;
title('Proposed');


subplot_tight(2, 5, 6,[0.01,.01]);
imagesc(Tensor_Y_Noiseless(:,:,frame2));
colormap(gray);axis image;axis off;

subplot_tight(2, 5, 7,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.I(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;


subplot_tight(2, 5, 8,[0.01,.01]);
imagesc(reshape(sub_infos_TeCPSGD.L(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;

subplot_tight(2, 5, 9,[0.01,.01]);
imagesc(reshape(sub_infos_olstec.L(:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;

subplot_tight(2, 5, 10,[0.01,.01]);
imagesc(reshape(sub_infos_brst.L(:,:,frame2),[rows,cols]));
colormap(gray);axis image;axis off;
%%
%save('video_completion_comparison')
