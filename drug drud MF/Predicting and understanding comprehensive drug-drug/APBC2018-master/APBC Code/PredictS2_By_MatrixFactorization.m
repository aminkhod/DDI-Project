function Predicted_LabelMat  = PredictS2_By_MatrixFactorization(trn_DDI_mat,TrnFeatureMat, tst_DDI_mat, TstFeatureMat,  option_)
% 
disp('Triple Matrix Factorization');
nTrn = size(TrnFeatureMat,1);
nTst = size(TstFeatureMat,1);
assert( nTrn == size(trn_DDI_mat,1) ) ;
assert( nTst == size(tst_DDI_mat,1) ) ;

assert( size(trn_DDI_mat,1) == size(trn_DDI_mat,2) ) ;
assert( nTrn == size(trn_DDI_mat,2) ) ;
assert(nTrn == size(tst_DDI_mat,2));
nComp = option_;



Predicted_LabelMat = PLSregression_S2(trn_DDI_mat,TrnFeatureMat,TstFeatureMat,nComp, false);

assert(sum( size(tst_DDI_mat)==size(Predicted_LabelMat)) ==2 );

%%
function Row_adj_predicted= PLSregression_S2(TrnAdj_S2,Fd_trn,Fd_tst,nComp, show)
% Fd_trn - 
% Fd_tst - 
if nargin < 5
    show =true;
end
%1 clearn: for the DTI matrix containing all-zero columns
TrnAdj_S2_New = TrnAdj_S2; % containing 0 and 1, or -1.

% clear training matrix
degree_row = sum(abs( TrnAdj_S2_New) ,2);  % ABS:  extend to find all-ZERO column
idx_Removed_Row= find(degree_row ==0);% S2 or S4 occuring in TRN
idx_Remaining_Row= find(degree_row >0); 
TrnAdj_S2_New(idx_Removed_Row,:)=[];

degree_col = sum(abs(TrnAdj_S2_New),1); % ABS:  extend to find all-ZERO column
idx_Removed_Col= find(degree_col ==0);  % S3 or S4 occuring in TRN
idx_Remaining_Col = find(degree_col >0);
TrnAdj_S2_New(:,idx_Removed_Col)=[];  

%% nmf,seminmf
% [W,H] = nmf(TrnAdj_S2_New', fix(rank(TrnAdj_S2_New)/2 ) ); % nComp , min( fix(rank(TrnAdj_S2_New)/2),nComp)
% [W,H] = nmf(TrnAdj_S2_New',nComp);   % nmf
[W,H] = semi_nmf(TrnAdj_S2_New',nComp);   % semi_nmf
Row_Response_trn = H'; %trn H as features
Col_Response_trn = W; %%W: TODO: for deep

%%
Row_Observation_trn= Fd_trn(idx_Remaining_Row,:);
Row_Observation_tst = Fd_tst;

threshold1 = nComp;

% PLS
% standard PLS:% Y_tst_pred= DoPLS(X_trn,X_tst,Y_trn,LV)
Row_Response_tst_predicted = DoPLS(Row_Observation_trn,Row_Observation_tst,Row_Response_trn,threshold1);

Row_adj_predicted  =Row_Response_tst_predicted * Col_Response_trn' ;% since RowView_Response_trn already contains sqrt(S_adj)


%% core funtion
function [Y_tst_pred, Beta_ ] = DoPLS(X_trn,X_tst,Y_trn,LV)
[~,~,~,~,Beta_]=plsregress( X_trn,Y_trn ,LV );
Y_tst_pred = ([ones(size(X_tst,1),1)  ,  X_tst]*Beta_);