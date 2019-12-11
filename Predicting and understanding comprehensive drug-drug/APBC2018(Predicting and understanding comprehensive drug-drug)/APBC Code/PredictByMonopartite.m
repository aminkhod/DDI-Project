function [ ave_AUC, ave_AUPR] =PredictByMonopartite...
    (DDI, FeatureMat, nCV, option_, IsBinary,Scenario,Debug)
% Adj: binary Adjacent matrix (0,1) between two differet nodes, such as lncRNA and proteins(genes)
%S_row,S_col: similarity matrices between lncRNAs and between genes.
% nCV  =10; nComp=6;
%
% DEMO: 
% PredictByMonopartite(DDI, Offsides_PCA,2 ,5, 'Binary','S4');
% PredictByMonopartite(DDI, Offsides_PCA,2 ,5, 'triple','S4');

if nargin <7
    Debug = true;
end
if nargin <6
    Scenario = 'S2';
end
if nargin <5
    IsBinary = 'Binary';
end

if nargin <4
    option_ = 1;
end

if nargin<3
    nCV =5;
end

fixRandom = Debug;
if fixRandom
    disp('Debug')
    rand('state',1234567); % fix random seed for debug
end

%% -----%%
%% ---- %%
switch upper(IsBinary)
    case 'BINARY'
    DDI(DDI~=0)=1; % only support binary prediction
    disp( 'running for binary interaction')
    
    case 'TRIPLE'
    disp( 'running for triple interaction')
    otherwise
        error('Wrong type of DDI')
end

disp(option_)
switch upper(Scenario)  %%NOTE: <Easy generated bias in S1>
    case 'S2'
        type='PGM'; % 'LP'; 'NV', 'RLS',  'SVM', 'TMF', 'PGM','RW'
        [ ave_AUC, ave_AUPR] = PerformS2(DDI, FeatureMat,nCV,option_, type);% for positive real-valued and binary
        
    case 'S4'
        [ ave_AUC, ave_AUPR] = PerformS4(DDI, FeatureMat,nCV,option_);% for positive real-valued and binary
        
    otherwise
        warning('Scenario is out of range')
end


%% : S2/S3
function [ave_AUC, ave_AUPR] = PerformS2(Adj, FeatureMat,nCV,option_,type)
[nRow, ~]=size(Adj);
CV_Idx_row= GenerateIdxForCV(nRow,nCV);
    
for k_fold = 1:nCV
    % split.
    CV_Temp= CV_Idx_row;    CV_Temp(k_fold) = [];
    ID_Trn_r =  cell2mat(CV_Temp) ;
    clear CV_Temp;
    ID_Tst_r =  CV_Idx_row{k_fold} ;
    
    TrnLabelMat= Adj(ID_Trn_r,ID_Trn_r); % may contain all-zero columns, which cause S4 prediction
    TstLabelMat = Adj(ID_Tst_r,ID_Trn_r);
    
    % core function
    switch upper(type)
        case 'LP'
            Sim_ = (FeatureMat); % using similarity matrix not feature matrix
            Predicted_LabelMat = PredictByLabelPropagation(Adj, Sim_, ID_Trn_r, ID_Tst_r, 0.1);   % 0.2         
        case 'RW'
            Sim_ = (FeatureMat); % using similarity matrix not feature matrix
            Predicted_LabelMat = PredictByRWR(Adj, Sim_, ID_Trn_r, ID_Tst_r, option_);
        case 'NV'
            Sim_ = (FeatureMat);% using similarity matrix not feature matrix
            Predicted_LabelMat= PredictByNaiveSimilarity_S2(TrnLabelMat,Sim_(ID_Tst_r,ID_Trn_r));
        case 'RLS'
            Sim_ = (FeatureMat);% using similarity matrix not feature matrix
            Predicted_LabelMat= PredictByRLS_S2(TrnLabelMat,Sim_(ID_Trn_r,ID_Trn_r),Sim_(ID_Tst_r,ID_Trn_r));
        case 'SVM'
            Sim_ = (FeatureMat);
            Predicted_LabelMat  = PredictS2_SVM(TrnLabelMat,Sim_(ID_Trn_r,ID_Trn_r), TstLabelMat, Sim_(ID_Tst_r,ID_Trn_r),  option_, true);
        case 'TMF' % using  feature matrix
            Predicted_LabelMat =PredictS2_By_MatrixFactorization...
                (TrnLabelMat,FeatureMat(ID_Trn_r,:), TstLabelMat, FeatureMat(ID_Tst_r,:),  option_);
		case 'PGM' % KuiTao's implementation
		    Predicted_LabelMat =PredictS2_By_MatrixFactorization...
                (TrnLabelMat,FeatureMat(ID_Trn_r,:), TstLabelMat, FeatureMat(ID_Tst_r,:),  option_);
%    Predicted_LabelMat=rand(size(TstLabelMat));
    end
    
    [AUC_(k_fold), AUPR_(k_fold) ] = Measure_S2(Predicted_LabelMat,TrnLabelMat,TstLabelMat, true);
    
    
end % CV ends

ave_AUC =mean(AUC_) ; ave_AUPR = mean(AUPR_);
disp([ave_AUC;ave_AUPR ]);
disp([std(AUC_); std(AUPR_)]);

%% : S4
function [ave_AUC, ave_AUPR] = PerformS4(Adj, FeatureMat,nCV,option_)
% Adj is a symmetric matrix

[nRow, nCol]=size(Adj);
CV_Idx_row= GenerateIdxForCV(nRow,nCV);
CV_Idx_col= CV_Idx_row;
OutputScores = zeros(size(Adj));

nComp = option_;
for k_fold_r = 1:nCV
    
    % find training entries of both interactions and non-interactions.
    % (1) firstly, find the training drugs and the testing drug to perform
    % S2 test.
    CV_Temp= CV_Idx_row;    CV_Temp(k_fold_r) = [];
    ID_Trn_r =  cell2mat(CV_Temp) ;     clear CV_Temp;
    ID_Tst_r =  CV_Idx_row{k_fold_r} ;
    
    % (2) assign the interactions between the testing targets in S4 and the
    % training drugs with ZEROS
    for k_fold_c = k_fold_r : nCV % considering the symmetry of DDI, 
        CV_Temp= CV_Idx_col;    CV_Temp(k_fold_c) = [];
        ID_Trn_c =  cell2mat(CV_Temp) ; clear CV_Temp;
        ID_Tst_c =  CV_Idx_col{k_fold_c} ;
        
        
        TrnLabelMat= Adj(ID_Trn_r,ID_Trn_c); % may contain all-zero columns, which cause S4 prediction
        TstLabelMat = Adj(ID_Tst_r,ID_Tst_c); % S4 block
        
        TrnFeatureMat_row= FeatureMat(ID_Trn_r,:);
        TstFeatureMat_row= FeatureMat(ID_Tst_r,:);
        TrnFeatureMat_col =  FeatureMat(ID_Trn_c,:);
        TstFeatureMat_col =  FeatureMat(ID_Tst_c,:);
        Predicted_LabelMat  = ...
            PredictS4_By_MatrixFactorization(...
            TrnLabelMat, TstLabelMat, ...
            TrnFeatureMat_row,TrnFeatureMat_col,...
    TstFeatureMat_row,TstFeatureMat_col,   nComp);
        
        
        [AUC_(k_fold_r,k_fold_c), AUPR_(k_fold_r,k_fold_c)]=Measure_S4(Predicted_LabelMat,TstLabelMat);
        OutputScores(ID_Tst_r,ID_Tst_c)     = Predicted_LabelMat;
    end
    
    
end
% since the left low triangle zone contains zeros entries.
AUC_entries = [squareform(AUC_'-diag(diag(AUC_)))' ; diag(AUC_)] ;
AUPR_entries = [squareform(AUPR_'-diag(diag(AUPR_)))' ; diag(AUPR_)] ;

ave_AUC =mean(AUC_entries ) ; ave_AUPR = mean(AUPR_entries);
disp([ave_AUC;ave_AUPR ]);
disp([std(AUC_entries); std(AUPR_entries)]);

%% : helper

%%
function [AUC_S2,AUPR_S2] = Measure_S2(Predicted_Scores,trn_Adj,tst_Adj, RemoveAllZeroTrn)
%% global measure
if nargin <4
    RemoveAllZeroTrn = false;
end
if RemoveAllZeroTrn
    threshold  =1; disp('only S2/S3');
else
    threshold  =0;
end


degrees_= sum(trn_Adj,1); % to remove S4 cases, if occured.
Scores  = Predicted_Scores(:,degrees_>=threshold);
Label_mat = tst_Adj(:,degrees_>=threshold);

TrueScore_PostiveClass =  Scores( Label_mat(:)>0) ; % LOG:change it to adapt -1,0,+1 triple-class
FalseScore= Scores( Label_mat(:)==0);
TrueScore_NegativeClass =  - Scores( Label_mat(:)<0) ; % LOG:change it to adapt -1,0,+1 triple-class

[AUC_S2, AUPR_S2 ]=EstimationAUC([TrueScore_PostiveClass;TrueScore_NegativeClass ],FalseScore,2000,0);

%
function [AUC_S4, AUPR_S4 ]=Measure_S4(Predicted_Scores,tst_Adj)
TrueScore_PostiveClass =  Predicted_Scores( tst_Adj(:)>0) ; % LOG:change it to adapt -1,0,+1 triple-class
FalseScore= Predicted_Scores( tst_Adj(:)==0);
TrueScore_NegativeClass =  - Predicted_Scores( tst_Adj(:)<0) ; % LOG:change it to adapt -1,0,+1 triple-class

[AUC_S4, AUPR_S4 ]=EstimationAUC([TrueScore_PostiveClass;TrueScore_NegativeClass ],FalseScore,2000,0);
