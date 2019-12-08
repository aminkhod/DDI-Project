function [Label,ProbV]=ClassIndifferent_Simple(DT,DP,FusionID,ClassiferID)
% Class Conscious fusion methods
% Decision Template...
% <input>:
%   DP          --  a cell contains a set of Decision Profile matrix
%                   {nSample}[nClassifer,nClass]
%   DT          --  a cell contains a set of Decision Template matrix
%                   {nClass}[nClassifer,nClass]
%   FusionID    --  The ID of fusion methods:'DT','DS','WDT'-WightedDT,'DSmT',
%   ClassiferID --  The classifers that are the parts of fusion shcema
%
% <output>:
%   Label   --  a set of continous class label discriminated by one fusion method[1 X nSample]
%   ProbV   --  a set of respective probability value of class label {nSample X 1}
%
% <See Also>:
%   ClassConscious.m  MajVoteLabelArray.m DecisionTemplates.m   ProbCell2DPCell.m
% <Remark>:
%   Decision Profile matrix is essential to fusion method which use the soft label
%   % Each class has a DT matrix，which is the mean of all the DP matrices in the class.
%
% Copyright 2002-2005,ICI reserves all rights.
% Author:(Shijianyu), snake5947@msn.com
% Created Time 01/12/2005,Last modified:01/13/2005
if nargin <4
    ClassiferID=[];
    if nargin <3
        FusionID='DT';
    end
end
% Preprocess to select classifiers.
[nClassifer,Dummy]=size(DP{1});
%if no selection, using all classifiers
if isempty(ClassiferID)
    ClassiferID=1:nClassifer;
end
for k=1:length(DT)
    DT{k}=DT{k}(ClassiferID,:);
end
for k=1:length(DP)
    DP{k}=DP{k}(ClassiferID,:);
end
%
% if isempty(DT)
%     disp('No DT used, so unsupervised')
% end

switch upper(FusionID)
    case 'DT'
        [Label,ProbV]=DTFusion(DT,DP);
    case 'DS'
        [Label,ProbV]=DSFusion(DT,DP);
        
    case 'DSMT' % only for binaray decision.
        [Label,ProbV]=DSmTFusion(DT,DP);
    otherwise
        error('Error Fusion ID-->''%s'' .\n ONLY ''DT'',''DS'' or ''DSMT'' is available !',FusionID);
end


%-----------------------DT----------------------------%
function [Label,Decision]=DTFusion(DT,DP)
%fprintf(1,'-> Decision Template Fusion !');%output on the screen
[nClassifer,nClass]=size(DP{1});
% 1 - DT: 进行Similarity Measure, Here only Euclidean norm is available
Decision=cell(length(DP),1);
for k=1:length(DP)%nSample
    Measure=zeros(1,length(DT));
    % 1.1 Euclidean Norm
    for n=1:length(DT)%nClass
        Measure(n) = 1- sum(sum(  ( DP{k}-DT{n} ).^2 /(nClassifer*nClass)  ));
    end
    % 1.2 Decision
    [ProbV(k),Label(k)]=max( Measure );
    Decision{k,1}=NormalizeV(Measure);% Soft decision
end

function NV=NormalizeV(V)
% <input>:
% V     --  Vector
% <output>:
% NV    --  Normalized Vector
if sum(V)==0
    NV = zeros(size(V));
    return;
end
SUM_V=sum(V)*ones(size(V));
NV=V./SUM_V;
%----------------------DST-----------------------------%
function [Label,Decision,ProbV]=DSFusion(DT,DP)
%fprintf(1,'-> Dempster-Shafer Fusion !\n');%output on the screen

[nClassifer,nClass]=size(DP{1});
K=1;% regular components
Decision=cell(length(DP),1);

Blief= cell(length(DP),1);
for S=1:length(DP)%nSample
    % 2.1 Proximity
    Proximity{S}=zeros(nClassifer,nClass);
    Blief{S}=zeros(nClassifer,nClass);
    
    EntropyVector=ShannonEntropy(DP{S});%CF, may be complex
    for L=1:nClassifer
        Prox = zeros(nClass,1);
        Sb = [];
        %%only for binary class
        DT_ave =  ( DT{1}(L,:)+DT{2}(L,:) ) /2;
        RelativeDist = DP{S}(L,:)- DT_ave +1 ;
        
        for C=1:nClass% length of DT
            
            PostProb = DP{S}(L,C) ;%1- sqrt( DP{S}(L,C)*EntropyVector(L) );
            %             PostProb= TransformProb(PostProb); % NOT GOOD
            %             DT{C}= TransformProb(DT{C}); % NOT GOOD
            
            if isempty(DT)
                Prox(C)=PostProb;%
            else
                % our new formula
                p=2;
                Ss =  1./(exp(norm(DT{C}(L,:)-DP{S}(L,:),2).^p ) ) ; % 1- proximity to the refernece point
                %                 Ss = 1./(1+(norm(DT{C}(L,:)-DP{S}(L,:),2).^2) ) ; % proximity to the refernece point
                % Prox(C)=PostProb;%
                
                if max( DT{C}(L,:)) <=0.8 % when the DT is not of high quality
                    Prox(C)=PostProb * Ss;%;Ss RelativeDist(C) ;%% p-norm. the original paper used 2-norm
                else
                    Prox(C)=PostProb * PostProb  ; %2 - PostProb
                end
                
                %                 Prox(C)=a*PostProb+(1-a)*1./(1+(norm(DT{C}(L,:)-DP{S}(L,:),2).^2) )    ;%% p-norm. the original paper used 2-norm
                %             Prox(C)= (1+norm(DT{C}(L,:)-DP{S}(L,:),p).^p) .^(-1)  ;%p-norm, the default is 1-norm
                %             Prox(C)= (1+norm(DT{C}(L,:)-DP{S}(L,:),'fro').^1) .^(-1)  ;%p-norm, the default is 1-norm
            end
        end
        if sum(Prox)==0
            disp('Complete Uncertainty found');
        else
            Prox=Prox./sum(Prox);
        end
        
        Proximity{S}(L,:)=Prox;
        % 2.2 blief: only for binary now
        Blf= zeros(1,nClass);
        for C=1:nClass
            Support = Prox(C);
            Conflict=ExclusiveProduct((1-Prox),C);
            Den = ( 1- Support*(1-Conflict) );
            Blf_pos = Support*Conflict;            Blf_all = (1-Support)*Conflict;            Blf_neg = 1-Blf_pos-Blf_all;
            Blf(C)=(Blf_pos)/Den;% original% %             Blf(C)=(Blf_pos+0.5*Blf_all)/Den;%  only for 2-class
            %             Blf(C)= Blf_pos/(Blf_pos+Blf_neg);
            %             Blf(C)= 1-(Blf_neg)/Den;
        end
        Blief{S}(L,:)=NormalizeV(Blf );% .* DT_ave; binary only  [DT{1}(L,1)-DT{2}(L,1)+1,DT{2}(L,2)-DT{1}(L,2)+1]
        %         Blief{S}(L,:) = Blief{S}(L,:)/sum(Blief{S}(L,:));
        
    end
    % 3 fusion
    FinalBlief = Blief{S} ;
    DT_w =  ( abs(DT{1} - DT{2})  .* [ abs( DT{1}(:,1)-DT{1}(:,2) ), abs( DT{2}(:,1)-DT{2}(:,2) ) ]  ); %(binary) separability as weights
    %     DT_w= 1./exp(DT_w);
    
    DT_w = DT_w./ repmat( sum(DT_w,1), size(FinalBlief,1) ,1);    % normalized along row (classifier)
    % %     DT_w = DT_w./ repmat( sum(DT_w,2), 1,size(FinalBlief,2) );    % normalized along col
    
    FinalBlief= FinalBlief .* ( DT_w); %3-Weighted
    
    Measure=median(FinalBlief,1);%median DS的处理    Measure=max(FinalBlief,[],1);%DS的处理
    
    Decision{S,1}=NormalizeV(Measure);% Soft decision
    [ProbV(S),Label(S)]=max( Decision{S,1} );%columnwise,也可以采用min,max,mean，prod
end
% newDecision = cell2mat(Decision);
%
function NewPostProb= TransformProb(PostProb)
% if PostProb>=0.5 && PostProb<=1
%     NewPostProb = 2*(PostProb-0.5).^2+0.5;
% elseif PostProb<0.5 && PostProb>=0
%     NewPostProb = -2*(PostProb-0.5).^2+0.5;
% else
%     error('PostProb is not a probability');
% end
% %2-
% if PostProb>=0.5 && PostProb<=1
%     NewPostProb = -2*(PostProb-1).^2+1;
% elseif PostProb<0.5 && PostProb>=0
%     NewPostProb = 2*(PostProb).^2;
% else
%     error('PostProb is not a probability');
% end
a= 0.55;
NewPostProb= PostProb;
for k=1:numel(PostProb)
    if PostProb(k)<=a && PostProb(k) >=(1-a)
        NewPostProb(k) = 1./(1+exp(-8*(PostProb(k)-0.5)));
    end
end


%%
function excProd=ExclusiveProduct(Vec,K)
% K -- the entry id not attending the multiplication.
excProd=1;
for n=1:length(Vec)
    if n==K
        continue;
    end
    excProd=excProd*Vec(n);
end

function EntropyVector=ShannonEntropy(ProbArray)
% Calculate Shannon Entropy of multiple classifiers for samples to measure cerntainty
% <Input>:
% ProbArray     --  each row is the output probabilities of classifiers [nClassifier X nClass]
% <Output>:
% EntropyVector --  column vector,normalized entropy vector[nClassifier X 1]
% <Remark>:
% ProbArray cannot contains all-zero row. the largger the Shannon Entropy is, the bigger the cerntainty[0,1] is
% <Reference>:
% Claude Shannon[]

[nClassifier,nClass]=size(ProbArray);


EntropyVector=zeros(nClassifier,1);
for k=1:nClassifier
    %     if ~isempty( find(ProbArray(k,:)<0) )
    %         ProbArray(k,:) = Normalize01(ProbArray(k,:));
    %     end
    
    for n=1:nClass
        p=ProbArray(k,n);
        if p==1
            EntropyVector(k)=0;break;
        end
        if p==0
            continue;
        end
        EntropyVector(k)=EntropyVector(k)+p*log2(p);
    end
end
EntropyVector=1 + EntropyVector/log2(nClass);% 归一化
% EntropyVector = abs(EntropyVector); %for complex

function NV=Normalize01(V)
%only for one negative and one positive entries
% map them into [0,1]
if numel(V)>2
    error('only support binary classification')
end

NV=V;
[max_v,idx_max ] = max(V);
[min_v,idx_min ] = min(V);

NV(idx_max) = NV(idx_max) / ( abs(max_v)+abs(min_v ) );
NV(idx_min) = 1- NV(idx_max) ;

%---------------------DSMT---------------------------------%
%Dezert-Smarandache Theory
function [Label,Decision]=DSmTFusion(DT,DP)
