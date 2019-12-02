function Predicted_Scores = DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, FusionName)
if nargin<5
    FusionName = 'DS';
end

Predicted_Scores = zeros(size(tst_DDI_mat));
for d= 1: size(trn_DDI_mat,2)
    % to make an adapter for DT DS fusion for multiple labels- column by
    % column
    Predicted_Self_Prob_Cell = cell (length(SelfPostProbInCol),1);
    Predicted_Prob_Cell = cell (length(PostProbInCol),1);
    if sum(trn_DDI_mat(:,d)) <1 % no positive samples
        prior_ = length(find(trn_DDI_mat(:)==1) ) / numel(trn_DDI_mat);
        Predicted_Scores(:,d) = prior_ * ones(size(tst_DDI_mat,1),1);
        disp(sprintf('No positive: %d', d));
        continue;
    end
    
    for p=1: length(Predicted_Prob_Cell)
        %     holder_trn = zeros(size(trn_DDI_mat(:,c) ) ) ;
        %      holder_tst = zeros(size(tst_DDI_mat(:,c) ) ) ;
        % [0, 1] : NOTE: since the class label will be ordered by their
        % values, put class 0 into the first position such that we get the
        % consistence with DecisionTemplates
        Predicted_Self_Prob_Cell{p,1}= [1-SelfPostProbInCol{p,1}(:,d),SelfPostProbInCol{p,1}(:,d),];
        Predicted_Prob_Cell{p,1}=  [1-PostProbInCol{p,1}(:,d), PostProbInCol{p,1}(:,d),]; %[0,1]
        
    end
    % TODO:refactoring
    DPCell = ProbCell2DPCell(Predicted_Prob_Cell);
    SelfDPCell = ProbCell2DPCell(Predicted_Self_Prob_Cell);
    
    [DTCell,ClassSampleCount,BaseClassLabels]=DecisionTemplates(trn_DDI_mat(:,d),SelfDPCell);
    
    [Label,ProbV]=ClassIndifferent_Simple(DTCell,DPCell,FusionName); % NOTE: something wrong in DSMT
    ProbV = cell2mat(ProbV);
    
%     [Label,ProbV]=DSmTDemo3(DTCell,DPCell,SelfDPCell,1:length(Predicted_Prob_Cell),trn_DDI_mat(:,d));
    
    Predicted_Scores(:,d)= ProbV(:,BaseClassLabels==1);
%     Label = Label-1;
end
% %% TODO
% Predicted_Scores = Predicted_Scores/ length(Predicted_Prob_Cell); % average fusion
disp(['our approaches------------Template fusion:',FusionName]);
