function [DTCell,ClassSampleCount,BaseClassLabels, ClassFrom,ClassTo]=DecisionTemplates(TrainClassID,TrainDP)
% Use the Decision Profile of the training data to construct the Decision
% Template of the training data.
% <input>:
%   TrainClassID    --  the class label of the training data(row vector)
%   TrainDP         --  a cell contains a set of Decision Profile matrix
%                       {nSample}[nClassifer,nClass]
% <output>:
%   DTCell          --  a cell contains a set of Decision Template matrix
%                       {nClass}[nClassifer,nClass]
% <See Also>:
%   ClassIndifferent.m  ProbCell2DPCell.m
%   DP- column vector£¬DT- row vector
% <Remark>:
%  Require that the samples are grouped by their classes in turn. [1,..,C]
% Copyright 2002-2005,ICI reserves all rights.
% Author:(Shijianyu), jianyushi@nwpu.edu.cn
% Created Time 01/12/2005,Last modified:08/03/2016

% Each class has a DT matrix£¬which is the mean of all the DP matrices in the class.
if size(TrainClassID,1)>1
    TrainClassID=TrainClassID';
end
% 0 - reorder samples according to their class labels in turn.
%[BaseClassLabels,BaseClassPosition,RepeaterCount]=FindClassBaseLabels(TrainClassID);
BaseClassLabels = unique(TrainClassID);
ClassSampleCount = zeros(size(BaseClassLabels));

NewTrainClassID=[];
NewTrainDP=[];
for k=1:length(BaseClassLabels)
    NewTrainClassID=[NewTrainClassID,TrainClassID( TrainClassID==BaseClassLabels(k) )];
    NewTrainDP=[NewTrainDP; TrainDP( TrainClassID==BaseClassLabels(k) ) ] ;
	
	ClassSampleCount(k) = sum(TrainClassID==BaseClassLabels(k));
end
TrainClassID=NewTrainClassID;
TrainDP=NewTrainDP;

% 1 - 
%[DummyAcc,mcc]=MCC(TrainClassID,TrainClassID);
%ClassSampleCount=mcc.pk;

ClassFrom=zeros(size(ClassSampleCount));
ClassTo=zeros(size(ClassSampleCount));
nClass=length(ClassSampleCount);

ClassFrom(1)=1;
%[Form,To)
for k=1:nClass
    ClassFrom(k+1)=ClassSampleCount(k)+ClassFrom(k);
    ClassTo(k)=ClassFrom(k+1);
end
ClassFrom(end)=[];% remove the last one
;
% 2 - Decision Template: average and max are good 
for k=1:nClass
    DTCell{k}=zeros( size(TrainDP{1}) );
    for n=ClassFrom(k):ClassTo(k)-1
        DTCell{k}=DTCell{k}+TrainDP{n}; %ave
%         DTCell{k}=max( DTCell{k},TrainDP{n});
    end
    DTCell{k}=DTCell{k}./ClassSampleCount(k); %ave
end

% %refactoring codes as follows: bugs in codes
% for k=1:nClass
%     DTCell{k}=zeros( size(TrainDP{1}) );
%     TrainDP_3D = zeros(size( TrainDP{1},1),size( TrainDP{1},2),  ClassTo(k)-ClassFrom(k));
%     for n=ClassFrom(k):ClassTo(k)-1
%         TrainDP_3D(:,:,n)=TrainDP{n}; %origninal
%     end
% % %     MAX is better than mean, when using RLS, or mean for SVM
%     DTCell{k}=mean(TrainDP_3D,3); %mean, median,
% %     DTCell{k}=max(TrainDP_3D,[],3); %max, min
% end

