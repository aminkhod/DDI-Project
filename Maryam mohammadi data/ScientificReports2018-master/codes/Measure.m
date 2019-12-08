function [ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores,trn_DDI_mat,tst_DDI_mat)
%% global measure

degrees_= sum(trn_DDI_mat,2); % to remove S4 cases, if occured.
Scores  = Predicted_Scores(:,degrees_>=1);
Label_mat = tst_DDI_mat(:,degrees_>=1);

TrueScore = Scores( Label_mat(:)==1);
FalseScore= Scores( Label_mat(:)~=1);

%
[AUC_S2, AUPR_S2 ]=EstimationAUC(TrueScore,FalseScore,2000,1);

%TODO:
% NC_global = ones(size(Label_mat,1),1);
% for k=1: size(Label_mat,1)
%     NC_global(k,1) = coverage(Scores(k,:)', Label_mat(k,:)' );
%     %NC_global is wrong 
% %     NC_global(k,1) = ( coverage(Scores(k,:)', Label_mat(k,:)' ) +1 - length(find(Label_mat(k,:)==1))  ) /   ( numel(Label_mat(k,:))  - length(find(Label_mat(k,:)==1)) );
% end

NC_global = coverage(Scores', Label_mat' );

ave_AUC= mean(AUC_S2);
if ave_AUC>1
    disp('');
end
ave_AUPR= mean(AUPR_S2);
ave_NC_global = mean(NC_global( ~isnan(NC_global) ) );
disp([ave_AUC,ave_AUPR,ave_NC_global])

%% for SVM and MLKNN
% PredictedLabel = Scores;
% PredictedLabel(PredictedLabel<0.5)=0;
% PredictedLabel(PredictedLabel>=0.5)=1;
% Coord = 1-xor(PredictedLabel,Label_mat); % #1/#all is just the accuracy


