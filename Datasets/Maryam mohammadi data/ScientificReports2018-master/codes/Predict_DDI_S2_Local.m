function [AUC,AUPR,NC_global,Predicted_Scores,tst_DDI_mat,AUC_Member_cell,AUPR_Member_cell] = ...
    Predict_DDI_S2_Local(DDI_Mat,DDI_Sim_Cell, k_percent, fixRandForDeBug, Method_ID)
%% Predict DDI for newly coming drugs
% DDI_Mat - adjacent matrix of drug-drug interactions, N x N
% DDI_Sim_Cell - one or more drug similarity matrices. N x N
%
% REMARK: since drugs must not interact with themselves, the diagnol of DDI_Mat are composed of ZEROS.
% if using kernel methods, remember to turn similarity matrix to PSD matrix
% by : lamda_=eig(Sd+0.00000001*eye(size(Sd) ));
% Using the correlation beween them, and set the Corr as weights
% corr([squareform(Sd_chem-diag(diag(Sd_chem)))', squareform(Sd_label-diag(diag(Sd_label)))' , squareform(Sd_off_label-diag(diag(Sd_off_label)))' , squareform(DDI-diag(diag(DDI)))'  ],'type','spearman')

% parameter setting
if nargin <5
    Method_ID = 'SINGLE';
end

if nargin < 4
    fixRandForDeBug = true;
end
if nargin<3
    k_percent = 0.15;
end

if fixRandForDeBug
    seed = 1234567890; % TODO: 50 repetitions: fix for debug only
    disp (seed)
    rand('state',seed); % for debug 1-90
end
disp(k_percent);

% (0) Initialization
for s=1:length(DDI_Sim_Cell)
    DDI_Sim_Cell{s}=  (DDI_Sim_Cell{s}+ DDI_Sim_Cell{s} )/2;
end
% DEFUALT: use the average as the default similarity matrix
Kd=zeros(size(DDI_Sim_Cell{1} ));
for s = 1:length(DDI_Sim_Cell)
    Kd= Kd + DDI_Sim_Cell{s};
end
Kd = Kd/length(DDI_Sim_Cell);

% % only for our approach.
% DDI_Mat = DDI_Mat -diag(diag(DDI_Mat))+ eye(size(DDI_Mat)); % if the entries on diagnol are zeros.

%% TODO: using two versions of cross validation, of which one is for the comparison with Ping Zhang et al. 's paper published on 2015
%% and another is for the general cross-validation
% (V1)%  -- use k% drugs for testing, %BIBM 2016 used
 [AUC,AUPR,NC_global,Predicted_Scores,tst_DDI_mat,AUC_Member_cell,AUPR_Member_cell] = HoldOutTest(DDI_Mat,DDI_Sim_Cell, Kd, k_percent, Method_ID);

%% for case study  % 4->35 444 448 551
% tst_idx_ = 4;
% trn_idx_ = 1:nDrug; trn_idx_(tst_idx_) =[];



function  [AUC,AUPR,NC_global,Predicted_Scores,tst_DDI_mat, AUC_Member_cell,AUPR_Member_cell] = HoldOutTest(DDI_Mat,DDI_Sim_Cell, Kd, k_percent,Method_ID)
nDrug = size(DDI_Mat,1);
idx_drug = randperm(nDrug);
% (1) split drugs randomly
tst_count = fix(k_percent  * nDrug);
tst_idx_ = sort( idx_drug(1 :  tst_count), 'ascend' );
trn_idx_ = sort( idx_drug(tst_count+1 :  end),  'ascend' );

% our approach
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);
AUC_Member_cell=[];AUPR_Member_cell=[];
disp(Method_ID)
switch upper(Method_ID)
    % (2) build local classification model
    case 'SINGLE'
        [AUC,AUPR,NC_global,Predicted_Scores] = PredictCoreByLocalModel_SingleSim(DDI_Mat,Kd, trn_idx_,tst_idx_);
    case 'SINGLE_FEA'    % Fd* Fd' cannot used as kernel, not PSD matrix
        Fd_cut= CancatenateSimilarity(DDI_Sim_Cell,  true);
%         Fd= CancatenateSimilarity(DDI_Sim_Cell,  false);
[AUC,AUPR,NC_global,Predicted_Scores] = ...
            PredictCoreByLocalModel_SingleFea(DDI_Mat,Fd_cut, trn_idx_,tst_idx_);

    case 'SIMPLE' % same classifier different similarity matrix
        [AUC,AUPR,NC_global,Predicted_Scores] = PredictCoreByLocalModel_MultiSim_SimplePostFusion(DDI_Mat,DDI_Sim_Cell, trn_idx_,tst_idx_);
        
    case 'DS' % Best
        [AUC,AUPR,NC_global,Predicted_Scores,AUC_Member_cell,AUPR_Member_cell,ave_pos_rank,ave_neg_rank,ComplexAcc] =...
            PredictCoreByLocalModel_MultiSim_PostFusion(DDI_Mat,DDI_Sim_Cell, trn_idx_,tst_idx_,false);
        
    case 'NAIVE'
        %other approaches
        % 2014 Nat
        [AUC,AUPR,NC_global,Predicted_Scores] =PredictCoreByNaive_MultiSim_PostFusion(DDI_Mat,DDI_Sim_Cell, trn_idx_,tst_idx_);
    case 'GLOBAL' % very very slow
        % 2014
        [AUC,AUPR,NC_global,Predicted_Scores] = PredictCoreByGlobalModel_MultiSim_Vectorize(DDI_Mat,DDI_Sim_Cell, trn_idx_,tst_idx_);
    case 'LP'
        [AUC,AUPR,NC_global,Predicted_Scores] =LabelPropagation(DDI_Mat,{Kd}, trn_idx_,tst_idx_);
    otherwise
        error('Wrong Method Name')
end
disp([AUC,AUPR,NC_global]);
disp('--->>NOTE: This function is closed on 2016-08-24');


%% sub-functions
%%
function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores] = PredictCoreByLocalModel_SingleSim(DDI_Mat,Kd, trn_idx_,tst_idx_)
tic %------------------->
trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
trn_Kd = Kd(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);
tst_Kd = Kd(tst_idx_,trn_idx_);

classifer_ = 'SVM';
Predicted_Scores = PredictBySimMat(trn_DDI_mat,tst_DDI_mat,trn_Kd,tst_Kd, classifer_); % 'SVM', 'MLKNN'
toc %------------------->
%% global measure
[ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);

function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores] = PredictCoreByLocalModel_SingleFea(DDI_Mat,Fd, trn_idx_,tst_idx_)
% better that average approach, but NOT good enough, compared with weighted similarity.
trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);

%%feature vector form
trn_Fd = Fd(trn_idx_,1:end); % trn_Fd = rand(size(trn_Fd));
tst_Fd = Fd(tst_idx_,1:end); % tst_Fd = rand(size(tst_Fd));
Predicted_Scores= zeros(size(tst_DDI_mat) );
for col = 1:size(trn_DDI_mat,2)
    if ~isempty(find(trn_DDI_mat(:,col)) == 1) % postive class
%         % direct SVM
option.C=4;option.g=0.005;
    Predicted_Scores(:,col)  = PredictCore_Feature_LibSVM(...
        trn_DDI_mat(:,col),tst_DDI_mat(:,col),...
        trn_Fd, tst_Fd,option);
    
    else % no positive in training
        Predicted_Scores(:,col) = ones(size(tst_DDI_mat,1),1) * length(find(trn_DDI_mat==1)) / numel(trn_DDI_mat);
    end
end
%% global measure
[ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);

%-3

%% simple fusion
function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores] = PredictCoreByLocalModel_MultiSim_SimplePostFusion(DDI_Mat,Kd_Cell, trn_idx_,tst_idx_)
% Our approach for multiple similarities/features
%
trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);

Predicted_Scores_ave = zeros(size(tst_DDI_mat));
Predicted_Scores_max = -10+zeros(size(tst_DDI_mat));
Predicted_Scores_prod = ones(size(tst_DDI_mat));

Predicted_Scores_W_ave = zeros(size(tst_DDI_mat));
w = [1; 1; 1];% 85% hold out, derived from Ping Zhang et al. Sci Rep 2015
if length(w) ~=length(Kd_Cell)
    error('#wieghts doesn''t match #similarities')
end
% -----
classifer_ ='SVM' ;
% disp(classifer_);
for p=1: length(Kd_Cell)
    
    trn_Kd = Kd_Cell{p}(trn_idx_,trn_idx_);
    tst_Kd = Kd_Cell{p}(tst_idx_,trn_idx_);
    
    
    PostProb = PredictBySimMat(trn_DDI_mat,tst_DDI_mat,trn_Kd,tst_Kd, classifer_); % 'SVM', 'MLKNN'
    Predicted_Scores_ave = Predicted_Scores_ave+PostProb; % for simple fusion only, avrage
    Predicted_Scores_max = max (Predicted_Scores_max, PostProb);
    
    Predicted_Scores_W_ave = Predicted_Scores_W_ave+PostProb*w(p);
    
    Predicted_Scores_prod = Predicted_Scores_prod .* PostProb;
end
%% simnple fusion
Predicted_Scores_ave = Predicted_Scores_ave/ length(Kd_Cell); % average fusion

Predicted_Scores_W_ave = Predicted_Scores_W_ave / sum(w);


disp('our approaches------------>average.');
Predicted_Scores = Predicted_Scores_ave;
%% global measure
[ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);


% ------------------------------------------------------------------------------------%
function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores, AUC_Member_cell,AUPR_Member_cell,ave_pos_rank,ave_neg_rank,ComplexAcc] = ...
                        PredictCoreByLocalModel_MultiSim_PostFusion(DDI_Mat,Kd_Cell, trn_idx_,tst_idx_, runAll)
% Our approach for multiple similarities/features
%
if nargin <5
    runAll= true;
end

trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);

option.C =1; %[0.125 0.25 0.5 1 2 4 8 16 32 64 128], no significant change
disp(option.C)


%% 1- using same type of classifiers, and individual similarty matrices: good for MCS
% [Vectorized_Predicted_Scores,Vectorized_Predicted_Scores_Self, PostProbInCol, SelfPostProbInCol,...
%     AUC_Member_cell,AUPR_Member_cell]=...
%             SameClassifiersWithIndividualSim(trn_DDI_mat,tst_DDI_mat,trn_idx_,tst_idx_,Kd_Cell,option);
        
%% 2-using different type of classifiers, and combined similarty matrix
option.classifier_  = { 'RLS' 'SVM' 'MLKNN' };
option.to_weight = true;
[Vectorized_Predicted_Scores,Vectorized_Predicted_Scores_Self, PostProbInCol, SelfPostProbInCol,...
    AUC_Member_cell,AUPR_Member_cell]=...
    DifferentClassifiersWithWeightedSim(trn_DDI_mat,tst_DDI_mat,trn_idx_,tst_idx_,Kd_Cell,option);

%% FUSION 
    disp('-------------Fusion----------------') % 
    % simple post fusion
    disp('simple post fusion');
    PosProbArray = zeros(size(tst_DDI_mat,1),size(tst_DDI_mat,2),length(PostProbInCol));
   for k=1: length(PostProbInCol)
       PosProbArray(:,:,k) = PostProbInCol{k};
   end
   
   %% <Insert Analysis codes of ranks >%
%    disp(tst_idx_);
%    PositiveRank =[ ...
%                             AnalyzeRank( PosProbArray(:,:,1), DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( PosProbArray(:,:,2), DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( PosProbArray(:,:,3), DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( sum(PosProbArray,3), DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( max(PosProbArray,[],3), DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( prod(PosProbArray,3),DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( min(PosProbArray,[],3),DDI_Mat,tst_idx_), ...
%                             AnalyzeRank( DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DT'),DDI_Mat,tst_idx_) , ...
%                             AnalyzeRank( DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DS'),DDI_Mat,tst_idx_), ...
%                             ];
%  Hitted_ = size(PositiveRank,1)-PositiveRank;
%  open PositiveRank;
%  open Hitted_;

% DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DS');
% 
ave_pos_rank=0; ave_neg_rank=0;

if runAll
    Predicted_Scores_Cell = {...
        PosProbArray(:,:,1); PosProbArray(:,:,2); PosProbArray(:,:,3);
        sum(PosProbArray,3) /length(PostProbInCol) ; max(PosProbArray,[],3);prod(PosProbArray,3) .^(1/length(PostProbInCol));  min(PosProbArray,[],3);
        DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DT');  DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DS') };
    
    [PositiveRank_arr, rank_arr,NegativeRank_arr] = AnalyzeRank_Cell(Predicted_Scores_Cell,DDI_Mat,tst_idx_);
    
    ave_pos_rank = mean(PositiveRank_arr,1);
    ave_neg_rank = mean(NegativeRank_arr,1);
    Predicted_Scores = Predicted_Scores_Cell{end};

else
    %DST fusion % BIBM used
    %%
    %    Predicted_Scores = sum(PosProbArray,3 /length(PostProbInCol)); % ave
    %    Predicted_Scores = max(PosProbArray,[],3); %
    %    Predicted_Scores = prod(PosProbArray,3).^(1/length(PostProbInCol)); %
    %     Predicted_Scores = min(PosProbArray,[],3); %
    
    % Max_prob = max(PosProbArray,[],3);
    % Min_prob = min(PosProbArray,[],3);
    DS_prob = DoFusion(trn_DDI_mat, tst_DDI_mat, SelfPostProbInCol, PostProbInCol, 'DS');
    Predicted_Scores = DS_prob;
end
    %% global measure <TODO>
    [ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);
    
ComplexAcc = [0,0,0];



%%
function [Vectorized_Predicted_Scores,Vectorized_Predicted_Scores_Self, PostProbInCol, SelfPostProbInCol,...
    AUC_Member_cell,AUPR_Member_cell]=...
            SameClassifiersWithIndividualSim(trn_DDI_mat,tst_DDI_mat,trn_idx_,tst_idx_,Kd_Cell,option)
        % for multi-layer MCS only
switch length(Kd_Cell)
    case 3
        Kd_Cell = [Kd_Cell, ...
            {(Kd_Cell{1}+Kd_Cell{2})/2},...% {sqrt(Kd_Cell{1}.*Kd_Cell{2})},...
            {(Kd_Cell{2}+Kd_Cell{3})/2},...%{sqrt(Kd_Cell{2}.*Kd_Cell{3})},...
            {(Kd_Cell{1}+Kd_Cell{3})/2},...%{sqrt(Kd_Cell{1}.*Kd_Cell{3})}, ...
            {(Kd_Cell{1}+Kd_Cell{2}+Kd_Cell{3})/3},...%{(Kd_Cell{1}.*Kd_Cell{2}.*Kd_Cell{2}).^(1/3) } ...
            ];
    case 2
        Kd_Cell = [Kd_Cell, {(Kd_Cell{1}+Kd_Cell{2})/2} ];
    otherwise
        disp('-->NOT support the combination in the case of  >3 similarities, just use no combination!! ')
        %DO NOTHING HERE
end

PostProbInCol = cell (length(Kd_Cell),1);
SelfPostProbInCol = cell (length(Kd_Cell),1);
Vectorized_Predicted_Scores =  zeros (numel(tst_DDI_mat),length(Kd_Cell));
Vectorized_Predicted_Scores_Self = zeros (numel(trn_DDI_mat),length(Kd_Cell));
classifier_type = 'SVM';
for p=1: length(Kd_Cell)
    
    trn_Kd = Kd_Cell{p}(trn_idx_,trn_idx_);
    tst_Kd = Kd_Cell{p}(tst_idx_,trn_idx_);
    % USE the similarity directly as output score to find the consistency
    % between similarity and interaction networks
    SelfPostProbInCol{p,1} =   PredictBySimMat(trn_DDI_mat,trn_DDI_mat,trn_Kd,trn_Kd, classifier_type,option); % 'SVM', 'MLKNN'
%     [AUC_self(p,1),AUPR_self(p,1),~] = Measure( SelfPostProbInCol{p,1}, trn_DDI_mat,trn_DDI_mat);

    PostProbInCol{p,1} = PredictBySimMat(trn_DDI_mat,tst_DDI_mat,trn_Kd,tst_Kd, classifier_type,option); % 'SVM', 'MLKNN' ,'RLS'
    [AUC_Member_cell{p},AUPR_Member_cell{p}]=Measure( PostProbInCol{p,1}, trn_DDI_mat,tst_DDI_mat);
    
    Vectorized_Predicted_Scores (:,p) = PostProbInCol{p,1}(:);
    Vectorized_Predicted_Scores_Self (:,p) = SelfPostProbInCol{p,1}(:);
end
disp('using same type of classifiers, and individual similarty matrices')

%%
function [Vectorized_Predicted_Scores,Vectorized_Predicted_Scores_Self, PostProbInCol, SelfPostProbInCol,...
                    AUC_Member_cell,AUPR_Member_cell]=...
    DifferentClassifiersWithWeightedSim(trn_DDI_mat,tst_DDI_mat,trn_idx_,tst_idx_,Kd_Cell,option)
classifer_ = option.classifier_;%{ 'SVM'; 'RLS'; 'MLKNN'};
PostProbInCol = cell (length(classifer_),1);
SelfPostProbInCol = cell (length(classifer_),1);
Vectorized_Predicted_Scores =  zeros (numel(tst_DDI_mat),length(Kd_Cell));
Vectorized_Predicted_Scores_Self = zeros (numel(trn_DDI_mat),length(Kd_Cell));

% % simple weighting
Combined_Kd = 0;
w= ones(length(Kd_Cell),1);
 if option.to_weight == true
     disp('Automatic weighting')
 end
for p=1: length(Kd_Cell)
    %% adapted weights by correlation
    corr_v=corr([squareform(Kd_Cell{p}(trn_idx_,trn_idx_)-diag(diag(Kd_Cell{p}(trn_idx_,trn_idx_) )))', ...
                    squareform(trn_DDI_mat-diag(diag(trn_DDI_mat)))'  ],'type','pearson');
                if option.to_weight == true                    
                    w(p,1)= abs(corr_v(1,2));
                end
    Combined_Kd =Combined_Kd + w(p) * Kd_Cell{p};
end
% Combined_Kd = Combined_Kd/ length(Kd_Cell);
Combined_Kd = Combined_Kd/ sum(w); % weighted
disp(w/ sum(w));

% % generalized weighted sum: can improve more
% for k=1:length(Kd_Cell)
%     SimMatCell {k,1}=  Kd_Cell{k}(trn_idx_,trn_idx_);
% end   
% [x_optim, fval_correlation]=MaxCombineSimilarity(SimMatCell, trn_DDI_mat)
% Combined_Kd = CombineSimilarityMatrices(Kd_Cell,x_optim);

for c=1: length(classifer_)
    
    trn_Kd = Combined_Kd(trn_idx_,trn_idx_);
    tst_Kd = Combined_Kd(tst_idx_,trn_idx_);
    
    PostProbInCol{c,1} =  PredictBySimMat(trn_DDI_mat,tst_DDI_mat,trn_Kd,tst_Kd, classifer_{c},option); % 'SVM', 'MLKNN' ,'RLS'
    [AUC_Member_cell{c},AUPR_Member_cell{c}]=Measure( PostProbInCol{c,1}, trn_DDI_mat,tst_DDI_mat);

    SelfPostProbInCol{c,1} = PredictBySimMat(trn_DDI_mat,trn_DDI_mat,trn_Kd,trn_Kd, classifer_{c},option); % 'SVM', 'MLKNN'
    
    Vectorized_Predicted_Scores (:,c) = PostProbInCol{c,1}(:);
    Vectorized_Predicted_Scores_Self (:,c) = SelfPostProbInCol{c,1}(:);

    disp('-----------------------------------')
end
disp('using different type of classifiers, and combined similarty matrix')
%% --------------------------------------------------------------------------------------------------------------------------------

%% Repeat other approaches
%-------------------------------------------------------------------------------------%
function [AUC,AUPR_Member_cell,NC_global,Predicted_Scores] = PredictCoreByGlobalModel_MultiSim_Vectorize(DDI_Mat,Kd_Cell, trn_idx_,tst_idx_)
%% The approach published in [Cheng F, et al. J Am Med Inform Assoc 2014;21:e278¨Ce286.]
% very slow because of many instances. 
tic % ------------------>
trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);


trn_Feature = zeros(( length(trn_idx_) * length(trn_idx_) - length(trn_idx_) )/2 , length(Kd_Cell) ); % only use the entries in the upper triangle, excluding the diagonal
tst_Feature = zeros(length(tst_idx_)* length(trn_idx_), length(Kd_Cell) );

trn_Labels = squareform( trn_DDI_mat )'; % ignore the order of entries; % only use the entries in the upper triangle, excluding the diagonal
tst_Labels = tst_DDI_mat(:);

for p=1: length(Kd_Cell)
    
    trn_Kd = Kd_Cell{p}(trn_idx_,trn_idx_);
    tst_Kd = Kd_Cell{p}(tst_idx_,trn_idx_);
    
    trn_Feature(:,p) = squareform( trn_Kd - diag(diag(trn_Kd)) )' ; % only use the entries in the upper triangle, excluding the diagonal
    tst_Feature(:,p) = tst_Kd(:);
    
    
end
%DOWNsampling to train
Cnt= 15000;
if length(trn_Labels) > Cnt% global classification is SLOW, can use downsampling
    idx= randperm( length(trn_Labels) );
    idx= idx(1:Cnt);
else
    idx = 1:length(trn_Labels);
end
%NEED to tune parameters of SVM.
Predicted_Scores= PredictCore_Feature_LibSVM(trn_Labels(idx),tst_Labels,trn_Feature(idx,:), tst_Feature); % very slow()

toc %-------------------<


IsShown = true;
[AUC, AUPR_Member_cell ]=EstimationAUC(Predicted_Scores(tst_Labels==1),Predicted_Scores(tst_Labels~=1),2000,IsShown);
NC_global = 0;
disp('Cheng F, et al. J Am Med Inform Assoc 2014 ---.')

%-------------------------------------------------------------------------------------%
function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores] = ...
    PredictCoreByNaive_MultiSim_PostFusion(DDI_Mat,Kd_Cell, trn_idx_,tst_idx_)
%% The approach published in [Vilar et al. Nat Protoc. 2014 September ; 9(9): 2147¨C2163.]
tic %------------------->


trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);

Predicted_Scores_Cell = cell (length(Kd_Cell),1);
Predicted_Scores = zeros(size(tst_DDI_mat));
Vectorized_Predicted_Scores =  zeros (numel(tst_DDI_mat),length(Kd_Cell));

for p=1: length(Predicted_Scores_Cell)
    
    trn_Kd = Kd_Cell{p}(trn_idx_,trn_idx_);
    tst_Kd = Kd_Cell{p}(tst_idx_,trn_idx_);
    
    Predicted_Scores_Cell{p,1}= tst_Kd* trn_DDI_mat  ;
    
    Predicted_Scores = Predicted_Scores+Predicted_Scores_Cell{p,1};
    
    Vectorized_Predicted_Scores (:,p) = Predicted_Scores_Cell{p,1}(:);
end
toc 
%% average fusion
% Predicted_Scores = Predicted_Scores/ length(Predicted_Scores_Cell); % average fusion
%% using PCA to combine
[trnPCA,~]= PCATransform(Vectorized_Predicted_Scores,[]);
Predicted_Scores = reshape( trnPCA(:,1), size(tst_DDI_mat) ); % the first PC

%-------------------<

%% global measure
[ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);
disp('Vilar et al. Nat Protoc. 2014---.')

%-------------------------------------------------------------------------------------%
function [ave_AUC,ave_AUPR,ave_NC_global,Predicted_Scores] =...
    LabelPropagation(DDI_Mat,Kd_Cell, trn_idx_,tst_idx_)
%% to known how to set the training and the testing
tic %-------------------< 
        % F = (1 ? miu)*(eye(size(W)) ? miu*W).^(-1)*Y
        Test_DDI_Mat = DDI_Mat;
        Test_DDI_Mat(Test_DDI_Mat~=1)= 0; % -1, or 0
        Test_DDI_Mat(tst_idx_,trn_idx_) = 0;
        miu = 0.25;
        W= Kd_Cell{end};
        %%should normalize by rows
        W = W ./ repmat( sum(W,2), 1,size(W,2)) ;
        
%         Predicted_Scores = (1-miu) *  (inv (   eye(size(W)) -  miu*W) * Test_DDI_Mat ) ; % slow way
        Predicted_Scores = (1-miu) *  ( (   eye(size(W)) -  miu*W) \ Test_DDI_Mat ) ; % fast way

        Predicted_Scores = Predicted_Scores(tst_idx_,trn_idx_);
        
trn_DDI_mat = DDI_Mat(trn_idx_,trn_idx_);
tst_DDI_mat = DDI_Mat(tst_idx_,trn_idx_);
toc %-------------------<

[ave_AUC,ave_AUPR,ave_NC_global] = Measure(Predicted_Scores.*1, trn_DDI_mat,tst_DDI_mat);
disp('Label Propagation. 2015---BUT very bad results are output. Need to figure out the reason.')
%% core functions
function Predicted_Scores = PredictBySimMat(trn_DDI_mat,tst_DDI_mat,trn_Kd,tst_Kd, classifierID, option)
% local model
if nargin<6
    option.C=1;
end
if nargin < 5
    classifierID = 'SVM';
end
Predicted_Scores = zeros(size(tst_DDI_mat) ) ;
disp(classifierID);

switch upper(classifierID)
    case 'SVM'
        C= option.C; %[0.125 0.25 0.5 1 2 4 8 16 32 64 128] training with few instances has bigger C
        prior_pos = (sum(trn_DDI_mat(:)==1)- size(trn_DDI_mat,1))/ numel(trn_DDI_mat(:));
        prior_neg = 1-prior_pos;%sum(trn_DDI_mat(:)~=1)/ numel(trn_DDI_mat(:));
        
        for d= 1: size(trn_DDI_mat,1)
            % all_idx_table = 1:length(trn_idx_);%all_idx_table ~=d
            CurrentTrnLabel = trn_DDI_mat(:,d);
            CurrentTstLabel = tst_DDI_mat(:,d);
            if isempty( find(CurrentTrnLabel==1) ) % no positive
                Predicted_Scores(:,d) = prior_pos * ones(size(CurrentTstLabel));
                disp(sprintf('No positive: %d', d));
                
            elseif isempty( find(CurrentTrnLabel~=1) ) % no negative
                Predicted_Scores(:,d) =1- prior_neg* ones(size(CurrentTstLabel));
                disp(sprintf('No negative: %d', d));
                
            else
                Predicted_Scores(:,d) = PredictCore_Feature_LibSVM_PreKernel(CurrentTrnLabel,CurrentTstLabel,trn_Kd, tst_Kd, C);
            end
        end
        
    case 'RLS'
        % % RLS
        TrnDDI_RLS = trn_DDI_mat;
        TrnDDI_RLS(TrnDDI_RLS~=1)= -1; % -1, or 0
        %     % Outputs_ = TstSimlarity*  inv (TrnSimlarity + 2* eye(size(TrnSimlarity)) ) *  TrnDTI_RLS ; % original slow way;
        Predicted_Scores = tst_Kd *  ( (trn_Kd + 0.5* eye(size(trn_Kd)) ) \ TrnDDI_RLS ) ; % fast way
        
        %TOOD: find a reasonable to normalize the values into probablity
%         Predicted_Scores(Predicted_Scores(:)<0)=0;
%         Predicted_Scores(Predicted_Scores(:)>1)=1;
        % map to [0,1] from [-1,+1] %% important to sqrt or multiplication!!
        Predicted_Scores = ( Predicted_Scores - min(Predicted_Scores(:)) ) ./ (max(Predicted_Scores(:)) - min(Predicted_Scores(:))); 
        if ~isempty( find( Predicted_Scores(:)<0 ) )  ||  ~isempty( find( Predicted_Scores(:)>1 ) ) 
            error('negative probabilty or unreasonable prob. found, need to be checked');
        end
        
    case 'MLKNN'
        % classifier 1 : KNN
        Num=5;Smooth=1; % default parameters and requirements of MLKNN
        TrnDDI_RLS = trn_DDI_mat;
        TstDDI_RLS = tst_DDI_mat;
        TrnDDI_RLS (TrnDDI_RLS~=1)  =-1;%required by MLKNN
        TstDDI_RLS (TstDDI_RLS~=1)  =-1;%required by MLKNN
        
        [Prior,PriorN,Cond,CondN]=MLKNN_trainWithKernel(trn_Kd,transpose(TrnDDI_RLS),Num,Smooth);%transpose for MLKNN
        [~,~,~,~,~,Predicted_Scores]...
            =MLKNN_testWithKernel(transpose(TrnDDI_RLS),tst_Kd,transpose(TstDDI_RLS),Num,Prior,PriorN,Cond,CondN);%transpose for MLKNN
        
        Predicted_Scores= Predicted_Scores'; %transpose
        

end

%% helper
function Fd= CancatenateSimilarity(K_row_cell,  RemoveFlag, dim)
if nargin <3
    dim =15;
end
if nargin <2
    RemoveFlag =false;
end
Fd= []; %ones(size(K_row_cell{1},1) );
for r=1:length(K_row_cell)
    f= SimMat2FeaMat(K_row_cell{r});
    if size(f,2)< size(f,1)
        f= [f,zeros(size(f,1), size(f,1)-size(f,2) )];
    end
%     disp(size(f,2))
    Fd=[Fd,f(:,1:dim) ];% concatenate features
%     Fd=[Fd.*f ];% concatenate features
end


%  delete columns contianing small values.
if RemoveFlag % remove redundency by PCA
    [Fd,~,~,~,~,flag]= PCATransform(Fd,Fd,[]);
    small_idx = find(flag<1e-6) ;
    if isempty(small_idx)
        The_end = length(flag);
    else
        The_end = small_idx(1)-1;
    end
    Fd= Fd(:,1:The_end);
end

%-
function F=SimMat2FeaMat(S)
[Ud,Sd,Vd]= svd(S);
[Ud,Sd,Vd,The_end_d]=CleanSVD(Ud,Sd,Vd);
F = Ud*sqrt(Sd);

%-
function [U_c,S_c,V_c,The_end]=CleanSVD(U,S,V)
% Remove small SVs

svd_entry = diag(S);
small_idx = find(svd_entry<1e-6) ;
if isempty(small_idx)
    The_end = length(svd_entry);
else
    The_end = small_idx(1)-1;
end

U_c= U(:,1:The_end);
S_c = S(1:The_end,1:The_end);
V_c = V(:,1:The_end);

%% Analysis
function [PositiveRank, rank_] = AnalyzeRank(Predicted_Scores,DDI_Mat,tst_idx_)
% tst_idx_ MUST be scalar
assert(isscalar(tst_idx_));

[SortedScores, IX] = sort( Predicted_Scores,'descend');
[~,rank_] = sort(IX,'ascend');

ConnectedIDx = find(DDI_Mat(tst_idx_ ,:));
greater_idx = ConnectedIDx > tst_idx_;

ConnectedIDx ( greater_idx  ) = ConnectedIDx ( greater_idx  ) -1;
PositiveRank= rank_(ConnectedIDx );
% disp(PositiveRank);

PositiveRank= PositiveRank';
rank_ = rank_';

%%
function [PositiveRank_arr, rank_arr,NegativeRank_arr] = AnalyzeRank_Cell(Predicted_Scores_Cell,DDI_Mat,tst_idx_)
% tst_idx_ MUST be scalar
assert(isscalar(tst_idx_));
PositiveRank_arr= [];
rank_arr =[];
NegativeRank_arr=[];
for k=1:length(Predicted_Scores_Cell)
    Predicted_Scores = Predicted_Scores_Cell{k};
    [PositiveRank, rank_] = AnalyzeRank(Predicted_Scores,DDI_Mat,tst_idx_);
    NegativeRank = setdiff(rank_, PositiveRank);
    
    PositiveRank_arr = [PositiveRank_arr, PositiveRank];
    rank_arr = [rank_arr, rank_];
    NegativeRank_arr = [NegativeRank_arr, NegativeRank];

end


