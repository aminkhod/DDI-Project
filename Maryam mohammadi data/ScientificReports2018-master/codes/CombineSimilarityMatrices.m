%%
function sum_mat = CombineSimilarityMatrices(SimMatCell,X,option)
%% Combine several similarity matrices by generalized weighted sum function
% X: w1, w2,....wn, k1,k2,...,kn
nCnt_x = length(X);
nCnt_sim = length(SimMatCell);
if option.useShift
    if  nCnt_sim * 3 ~=nCnt_x
        error('unmatched.');
    end
else
    if nCnt_sim * 2 ~=nCnt_x
        error('unmatched.');
    end
end
% combine similarity matrices by generalized weighted sum
sum_mat = 0;
for s =1:nCnt_sim
    w=  X(s+nCnt_sim );
    sim_ = SimMatCell{s};
    if option.useShift
        offset = X(s+2*nCnt_sim );
    else
        offset=mean( squareform(sim_- diag(diag(sim_))  )  );
    end
    
    %logistic function as the mapping
    Transformer = 1./(1+exp(- w * (sim_ - offset ) )  );
    
    sum_mat = sum_mat + X(s)* Transformer; % w* transformer
end
% sum_mat = sum_mat/ sum(X(1:nCnt_sim));