function out = streaming_bayesian_completion(stream,sample_tensors,forgetting_factor,sliding_window_size,ml_initialization,max_rank)
%streaming_bayesian_completion(stream,sample_tensors,forgetting_factor,sliding_window_size,ml_initialization,max_rank)
%stream = cell of complete tensors
%sample_tensors = cell of sampled entries
%forgetting_factor = exponential weighting
%sliding_window_size = how far to look back
%ml_initialization = if 1, initialize via SVD

stream_length = length(stream);

out = cell(2,stream_length); %first row is factors, second is removed errors

stream_with_errors_removed = cell(1,stream_length);

first_model = BayesRCP_TC(stream{1,1},'obs',sample_tensors{1},'verbose',0,'maxRank', max_rank);
temp = ['Completed step',' ',num2str(1)];
disp(temp)

out{1,1} = first_model.Z;
out{2,1} = first_model.E;
stream_with_errors_removed{1} = stream{1}-out{2,1};

for frame_number = 2:stream_length
    
    [out{1,frame_number},out{2,frame_number}] = bayesian_streaming_update_revamped(stream{1,frame_number},out{1,frame_number-1}',sample_tensors{frame_number});
    stream_with_errors_removed{frame_number} = stream{frame_number}-out{2,frame_number};
    
    temp = ['Completed step',' ',num2str(frame_number)];
    disp(temp)
    
end


    function [factors_out, errors_out] = bayesian_streaming_update_revamped(Y,input_factors,sampled_entries)
        
        dimY = size(Y);
        lowest_index_bound = max([1,frame_number-sliding_window_size+1]);
        num_slices = frame_number-lowest_index_bound+1;
        dimFull = [dimY,num_slices];
        
        N = length(dimY);
        O     = sampled_entries;
        R   = max_rank;
        maxiters  = 500;
        tol   = 1e-4;
        DIMRED   = 0;
        initVar   = 1;
        updateHyper = 'on';
        predVar = 0;
        randn('state',1); rand('state',1); %#ok<RAND>
        
        O = tensor(O);
        nObs = sum(O(:));
        LB = 0;
        
        a_lambda0     = 1e-6;
        b_lambda0     = 1e-6;
        a_tau0      = 1e-6;
        b_tau0      = 1e-6;
        a_gamma0     = 1e-6;
        b_gamma0     = 1e-6;
        
        lambdas = (a_lambda0+eps)/(b_lambda0+eps)*ones(R,1);
        tau = (a_tau0+eps)/(b_tau0+eps);
        gammas = initVar.^(-1)*ones(dimY).*((a_gamma0+eps)/(b_gamma0+eps));
        
        E = gammas.^(-0.5).*randn(dimY);
        Sigma_E = gammas.^(-1).*ones(dimY);
        
        Z = cell(N+1,1);
        ZSigma = cell(N+1,1);
        
        %compute weighted sum and normalizing constant for
        %forgetting factor sum in line 148
        
        %return here since initialization is a bit sketchy
        init_tensor = zeros(dimFull);
        samples_init_tensor = zeros(dimFull);
        test_tensor = zeros(dimFull);
        idx = repmat({':'}, 1, length(dimY));

        for i = 1:frame_number-lowest_index_bound+1
            init_tensor(idx{:},i) = stream{lowest_index_bound+i-1}.*sample_tensors{lowest_index_bound+i-1};
            samples_init_tensor(idx{:},i) = sample_tensors{lowest_index_bound+i-1};
            test_tensor(idx{:},i) = stream{lowest_index_bound+i-1};
        end
%         total_weight = 1;
%         weighted_sum= Y.*O;
%         
%         for i = (frame_number-1):-1:lowest_index_bound
%             
%             exponent = (frame_number-1) -i ;
%             weighted_sum = weighted_sum+(forgetting_factor^exponent)*stream_with_errors_removed{i}.*O;
%             total_weight = total_weight+forgetting_factor^exponent;
%             
%         end
%         
%         init_tensor = weighted_sum/total_weight;



    if ~isempty(find(O==0))
        init_tensor(find(O==0)) = sum(init_tensor(:))/nObs;
    end
    for n = 1:(N+1)
        ZSigma{n} = (repmat(diag(lambdas.^(-1)), [1 1 dimFull(n)]));
        [U, S, V] = svd(double(tenmat(init_tensor,n)), 'econ');
        if R <= size(U,2)
            Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
        else
            Z{n} = [U*(S.^(0.5)) randn(dimFull(n), R-size(U,2)).*((a_lambda0/b_lambda0).^(-0.5))];
        end
    end
    Y = Y.*O;

        %% Model learning
        
        % --------- E(aa') = cov(a,a) + E(a)E(a')----------------
        EZZT = cell(N+1,1);
        for n=1:N
            EZZT{n} = (reshape(ZSigma{n}, [R*R, dimFull(n)]))' + khatrirao_fast(Z{n}',Z{n}')';
        end
        
        EZZT_time = cell(num_slices,1);
        
        for k=1:num_slices
            EZZT_time{k} = (reshape(ZSigma{N+1}(:,:,k), [R*R, 1]))' + khatrirao_fast(Z{N+1}(k,:)',Z{N+1}(k,:)')';
        end
        
        for it=1:maxiters
            %% Update factor matrices
            Aw = diag(lambdas);
            
            
           
            for n=1:N
               %  norm_check()
                % compute E(Z_{\n}^{T} Z_{\n})
                %         Eslash = khatrirao(EZZT{[1:n-1, n+1:N]},'r');
                
                temp_kr = khatrirao_fast(EZZT{[1:n-1, n+1:N]},'r'); 
                temp_kr = kr(temp_kr,EZZT_time{num_slices})';
                                
                ENZZT = reshape(temp_kr * double(tenmat(O,n)'), [R,R,dimFull(n)]);
                 
                ENZZT_prev = cell(1,frame_number-lowest_index_bound);
                
                for i = 1:(num_slices-1)
                    temp_kr = khatrirao_fast(EZZT{[1:n-1, n+1:N]},'r');
                    temp_kr = kr(temp_kr,EZZT_time{i})';                                       
                    ENZZT_prev{i} = reshape(temp_kr * double(tenmat(sample_tensors{i+lowest_index_bound-1},n)'), [R,R,dimFull(n)]);
                  % cond(ENZZT_prev{i}(:,:,1))
                end

                temp_kr = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
                temp_kr = kr(temp_kr,Z{N+1}(num_slices,:))';
                
                FslashY = temp_kr * tenmat((Y-E).*O, n)';
                
                for j =  (num_slices-1):-1:1
                    temp_kr = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
                    temp_kr = kr(temp_kr,Z{N+1}(j,:))';
                    
                    FslashY = FslashY+(forgetting_factor^(num_slices-j))*temp_kr* tenmat((stream_with_errors_removed{lowest_index_bound-1+j}).*sample_tensors{lowest_index_bound-1+j}, n)';
                    %cond(double(FslashY))
                end
                
                for i=1:dimFull(n)
                    %build appropriate sum
                    ENZZT_sum = ENZZT(:,:,i);
                    for j = (frame_number-1):-1:lowest_index_bound
                        ENZZT_sum = ENZZT_sum+(forgetting_factor^(frame_number-j))*ENZZT_prev{j-lowest_index_bound+1}(:,:,i);
                    end
                    
                    ZSigma{n}(:,:,i) = (tau * ENZZT_sum + Aw)^(-1);
                    Z{n}(i,:) = (tau * ZSigma{n}(:,:,i) * FslashY(:,i))';
                end
                
                EZZT{n} = (reshape(ZSigma{n}, [R*R, dimFull(n)]) + khatrirao_fast(Z{n}',Z{n}'))';
            end
            
            
            %%Update Time Matrices
            
            %build big sample tensor
            idx = repmat({':'}, 1, length(dimY));
            big_sample_tensor = zeros(dimFull);
            
            for i = frame_number-lowest_index_bound+1:-1:1
                big_sample_tensor(idx{:},i) = sample_tensors{lowest_index_bound+i-1};
            end
                
            ENZZT = reshape(khatrirao_fast(EZZT{[1:N]},'r')'* double(tenmat(big_sample_tensor,n+1)'), [R,R,dimFull(n+1)]);
            
            temp_kr = khatrirao_fast(Z{[1:N]},'r')';
            
            for k=1:num_slices
                % compute E(Z_{\n}^{T} Z_{\n})
                %         Eslash = khatrirao(EZZT{[1:n-1, n+1:N]},'r');                    
                                
                if k == num_slices %only include errors on current term    
                    temp_vec = (Y-E).*O;
                    FslashY = temp_kr * temp_vec(:);
                else
                    temp_vec = stream_with_errors_removed{lowest_index_bound+k-1}.*sample_tensors{lowest_index_bound+k-1};
                    FslashY = temp_kr * temp_vec(:);                    
                end
                
                ZSigma{N+1}(:,:,k) = (tau*(forgetting_factor^(num_slices-k)) * ENZZT(:,:,k) + Aw)^(-1);
                Z{N+1}(k,:) = (tau * (forgetting_factor^(num_slices-k)) * ZSigma{N+1}(:,:,k) * FslashY(:))';
                
           
              %  EZZT{N+1} = (reshape(ZSigma{N+1}, [R*R, dimFull(N+1)]) + khatrirao_fast(Z{N+1}',Z{N+1}'))';
            end
            for k=1:num_slices
                EZZT_time{k} = (reshape(ZSigma{N+1}(:,:,k), [R*R, 1]))' + khatrirao_fast(Z{N+1}(k,:)',Z{N+1}(k,:)')';
            end
        
            %% Update latent tensor X %done this
            tempZ = Z(1:N);
            tempZ{N+1} = Z{N+1}(end,:); 
            X = double(ktensor(tempZ));
            
            %% Update hyperparameters lambda  %done this
            a_lambdaN = (0.5*sum(dimY)+1 + a_lambda0)*ones(R,1);
            b_lambdaN = 0;
            for n=1:N+1
                b_lambdaN = b_lambdaN + diag(Z{n}'*Z{n}) + diag(sum(ZSigma{n},3));
            end
            b_lambdaN = b_lambda0 + 0.5.* b_lambdaN;
            lambdas = a_lambdaN./b_lambdaN;
            
            %% update noise tau  %this update is next
            %  The most time and space consuming part
            
            
           if 1 % save time but large space needed
           EZZT_temp = EZZT;
           EZZT_temp{N+1} = EZZT_time{num_slices};
           EX2 =  O(:)' * khatrirao_fast(EZZT_temp,'r') * ones(R*R,1);
           else  % save space but slow
                temp1 = cell(N+1,1);%here
                EX2 =0;
                for i =1:R
                    for n=1:N+1 %HERE
                        if n == N+1
                            temp1{n} = EZZT_time{num_slices}(:,(i-1)*R+1: i*R);  
                        else
                            temp1{n} = EZZT{n}(:,(i-1)*R+1: i*R);
                        end
                    end
                    EX2 = EX2 + O(:)' * khatrirao(temp1,'r')* ones(R,1);
                end
            end
            
            %HERE
            EE2 = sum((E(:).^2 + Sigma_E(:)).*O(:));
            
            err = Y(:)'*Y(:) - 2*Y(:)'*X(:) -2*Y(:)'*E(:) + 2*X(:)'*E(:) + EX2 + EE2;
            
            
            total_window_obs = sum(samples_init_tensor(:));
            a_tauN = a_tau0 + 0.5*nObs;
            b_tauN = b_tau0 + 0.5*err;
            tau = a_tauN/b_tauN;
            
            %% Update sparse matrix E %done this
            Sigma_E = double(O)./(gammas+tau);
            E = double(tau*Sigma_E.*(Y-X).*O);
            
            %% Update the gammas %done this
            
            a_gammaN = a_gamma0 + 0.5*(double(O));
            b_gammaN = b_gamma0 + 0.5*(E.^2 + Sigma_E);
            gammas = a_gammaN./b_gammaN;
               
            %% Lower bound
            temp1 = -0.5*nObs*safelog(2*pi) + 0.5*nObs*(psi(a_tauN)-safelog(b_tauN)) - 0.5*(a_tauN/b_tauN)*err;
            temp2 =0;
            for n=1:N %here
                temp2 = temp2 + -0.5*R*dimY(n)*safelog(2*pi) + 0.5*dimY(n)*sum(psi(a_lambdaN)-safelog(b_lambdaN)) -0.5*trace(diag(lambdas)*sum(ZSigma{n},3)) -0.5*trace(diag(lambdas)*Z{n}'*Z{n});
            end
                temp2 = temp2 + -0.5*R*safelog(2*pi) + 0.5*sum(psi(a_lambdaN)-safelog(b_lambdaN)) -0.5*trace(diag(lambdas)*sum(ZSigma{N+1}(:,:,num_slices),3)) -0.5*trace(diag(lambdas)*Z{N+1}(num_slices,:)'*Z{N+1}(num_slices,:));
            
            temp3 = sum(-safelog(gamma(a_lambda0)) + a_lambda0*safelog(b_lambda0) -  b_lambda0.*(a_lambdaN./b_lambdaN) + (a_lambda0-1).*(psi(a_lambdaN)-safelog(b_lambdaN)));
            temp4 = -safelog(gamma(a_tau0)) + a_tau0*safelog(b_tau0) + (a_tau0-1)*(psi(a_tauN)-safelog(b_tauN)) - b_tau0*(a_tauN/b_tauN);
            temp5 = 0.5*R*sum(dimY)*(1+safelog(2*pi));
            for n=1:N+1
                if n == N+1
                    temp5 = temp5 + 0.5*safelog(det(ZSigma{N+1}(:,:,num_slices))) ;
                else
                    for i=1:size(ZSigma{n},3)
                        temp5 = temp5 + 0.5*safelog(det(ZSigma{n}(:,:,i))) ;
                    end
                end
            end
            temp6 = sum(safelog(gamma(a_lambdaN)) - (a_lambdaN-1).*psi(a_lambdaN) -safelog(b_lambdaN) + a_lambdaN);
            temp7 = safelog(gamma(a_tauN)) - (a_tauN-1)*psi(a_tauN) -safelog(b_tauN) + a_tauN;
            
            temp = psi(a_gammaN) - safelog(b_gammaN);
            temp8 = -0.5*nObs*safelog(2*pi) + 0.5*(temp(:)'*O(:)) - 0.5*(E(:).^2 + Sigma_E(:))'*gammas(:);
            temp9 = -nObs*safelog(gamma(a_gamma0)) + nObs*a_gamma0*safelog(b_gamma0) + ((a_gamma0-1)*(temp(:)) - b_gamma0*gammas(:))'*O(:);
            
            temp10 = 0.5*sum(safelog(Sigma_E(:))) + 0.5*nObs*(1+safelog(2*pi));
            temp11 = safelog(gamma(a_gammaN)) - (a_gammaN-1).*psi(a_gammaN) -safelog(b_gammaN) + a_gammaN;
            temp11 = temp11(:)'*O(:);
            
            LB(it) = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8 + temp9 + temp10 + temp11;
%             
%             %% update top level hyperparameters
%             if strcmp( updateHyper, 'on')
%                 if it>5
%                     aMean = gammas(:)'*O(:)/nObs;
%                     bMean = (psi(a_gammaN(:)) - safelog(b_gammaN(:)))'*O(:)/nObs;
%                     ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
%                     a_gamma0 = fminsearch(ngLB,a_gamma0);
%                     %       a_gamma0 = 0.55/(log(aMean)-bMean);
%                     b_gamma0 = a_gamma0/aMean;
%                     % % % % % % %       ngLB = @(x) log(gamma(x(1))) - x(1).*log(x(2)) - (x(1)-1)*bMean + x(2)*aMean;
%                     
% %                     % update the top level noise parameter
% %                     aMean = tau;
% %                     bMean = psi(a_tauN) - safelog(a_tauN);
% %                     ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
% %                     a_tau0 = fminsearch(ngLB,a_tau0);
% %                     b_tau0 = a_tau0/aMean;
% %                     
% %                     % update the top level gamma parameter
% %                     aMean = mean(gammas);
% %                     bMean = mean(psi(a_gammaN) - safelog(b_gammaN));
% %                     ngLB = @(x) log(gamma(x)) - x.*log(x/aMean) - (x-1)*bMean + x;
% %                     a_gamma0 = fminsearch(ngLB,a_gamma0);
% %                     b_gamma0 = a_gamma0/aMean;
%                 end
%             end
            
            %% Prune irrelevant dimensions?
            Zall = cell2mat(Z);
            comPower = diag(Zall' * Zall);
            comTol = sum(dimY)*eps(norm(Zall,'fro'));
            rankest = sum(comPower> comTol );
            if max(rankest)==0
                error('Rank becomes 0');
            end
            if DIMRED==1  && it >=2,
                if R~= max(rankest)
                    indices = comPower > comTol;
                    lambdas = lambdas(indices);
                    temp = ones(R,R);
                    temp(indices,indices) = 0;
                    temp = temp(:);
                    for n=1:N+1
                        Z{n} = Z{n}(:,indices);
                        ZSigma{n} = ZSigma{n}(indices,indices,:);
                        EZZT{n} = EZZT{n}(:, temp == 0);
                    end
                    R = rankest;
                end
            end
            
            %% Display progress
            
            if it>2
                converge = -1*((LB(it) - LB(it-1))/LB(2));
            else
                converge =inf;
            end
             
            %% Convergence check
            
            if it>5 && abs(converge)< tol
                %disp('\\\======= Converged===========\\\');
                
%                 disp("Number of iterations until convergence");
%                 it
%                 disp("Difference was ")
%                 norm(it_new(:)-it_old(:))/norm(it_old(:))
                break;
            end

        end
        
        %% Predictive distribution
%         switch predVar
%             case 1
%                 Xvar =  tenzeros(size(Y));
%                 for n=1:N
%                     Xvar = tenmat(Xvar,n);
%                     Fslash = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
%                     if 1
%                         temp1 = double(tenmat(tensor(ZSigma{n}),3));
%                         temp2 = khatrirao_fast(Fslash', Fslash');
%                         Xvar(:,:) = Xvar(:,:) + temp1*temp2;
%                     else
%                         % ---  slow computation ------
%                         for i=1:size(Xvar,1)     %#ok
%                             Xvar(i,:) = Xvar(i,:) + diag(Fslash * ZSigma{n}(:,:,i) *Fslash')';
%                         end
%                         % ---  slow computation ------
%                     end
%                     Xvar = tensor(Xvar);
%                 end
%                 Xvar = Xvar + tau^(-1);
%                 Xvar = Xvar.*(2*a_tauN)/(2*a_tauN-2);
%             case 2
%                 % Better for saving memory
%                 Xvar = double(ktensor(EZZT))- X.^2;
%                 Xvar = Xvar + tau^(-1);
%             otherwise
%                 Xvar =[];
%         end
        
        %% Output
        %         model.Z = Z;
        %         model.ZSigma = ZSigma;
        %         model.gammas = gammas;
        %         model.E = E;
        %         model.Sigma_E = Sigma_E;
        %         model.tau = tau;
        %         model.Xvar = double(Xvar);
        %         model.TrueRank = rankest;
        %         model.LowBound = max(LB);
        
        factors_out = cell(1,N+1);
        factors_out(1:N) = Z(1:N);
        factors_out(N+1) = {Z{N+1}(num_slices,:)};
        % error_cutoff = 2*std(E(:));
        % E(E<error_cutoff) = 0;
        errors_out = E;
        
        function y = safelog(x)
            x(x<1e-300)=1e-200;
            x(x>1e300)=1e300;
            y=log(x);
            
        end
        function norm_check()
            
            temp = zeros(dimFull);
            temp(:,:,:,num_slices) = E;
            
            norm(sum(khatrirao_fast(Z,'r'),2)-test_tensor(:)-temp(:))/norm(test_tensor(:))
        end
    end
end
