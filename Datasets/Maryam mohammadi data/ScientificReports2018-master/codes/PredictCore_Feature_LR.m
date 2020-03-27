 function [Outputs_ ] = PredictCore_Feature_LR(TrnInteraction,Feature_Mat_trn, Feature_Mat_tst)
 % for local model specific to each columns in TrnInteraction
X_trn = Feature_Mat_trn;
X_tst =Feature_Mat_tst ;
y_trn = TrnInteraction;
y_trn(y_trn~=1)= 0;

%
B = zeros(size(X_trn,2)+1, size(y_trn,2));
for k=1: size(y_trn,2)
    B(:,k) = glmfit(X_trn,y_trn(:,k),'binomial'); %logit: log(u/(1-u)) = Xb=yscore;  
end
% DEBUG only: use training
% ytr_score = [ones(size(X_trn,1),1),X_trn] * B;
% Outputs_trn = 1./(1+exp(-ytr_score) );

yst_score = [ones(size(X_tst,1),1),X_tst] * B; % ASSERT that it has the same dimension as TstDTI
Outputs_= 1./(1+exp(-yst_score) ); % use this form to prevent over-flow
% Outputs_= exp(yst_score)./(1+exp(yst_score) ) ; %inf/(1+inf) this form will cause infinite value, cannot be used
% Outputs_ = yst_score;  