function [x_optim, fval]=MaxCombineSimilarity(SimMatCell, DDI,option)
%% solving optimization:
%% Combine several similarity matrices optimally by generalized weighted sum function
% Obj: corr2( sum( wi * map(Sim) ), DDI )
% s.t.: according to logisitc function 1/(1+exp(-k(x-t)) )
% X: w1, w2,....wn, k1,k2,...,kn,t1,t2,...tn

useShift =option.useShift;
t_0=[];
lb_t=[];
ub_t=[];
Aeq_t=[];
if useShift
%     for p=1:length(SimMatCell)
%         t_0(p,1) = mean( squareform(SimMatCell{p}- diag(diag(SimMatCell{p} ))  )  );
%     end;
    
    t_0 = zeros(length(SimMatCell),1);
    lb_t = -1*ones(length(SimMatCell),1);
    ub_t = ones(length(SimMatCell),1);
    Aeq_t = zeros(length(SimMatCell),1)' ;
end


X0 = [rand(length(SimMatCell),1)/ length(SimMatCell); ... % weights
    ones(length(SimMatCell),1);... % steepness in logistic function
    t_0];

% # %  such that (s.t.) TODO: using three parameters
K  = 6;
lb =[zeros(length(SimMatCell),1);0*ones(length(SimMatCell),1); lb_t ];
ub= [ones(length(SimMatCell),1); +K*ones(length(SimMatCell),1); ub_t ];

Aeq =  [ones(length(SimMatCell),1)' , zeros(length(SimMatCell),1)'  ,Aeq_t ]  ;
beq = [1];

% solving obj with s.t.
options  = optimoptions('fmincon', 'Algorithm','trust-region-reflective'); %'active-set' 'sqp' 'trust-region-reflective'
[x_optim,fval] = fmincon(@(X) Corr2_(SimMatCell, DDI,useShift,X),X0, [],[],Aeq,beq, lb,ub,[],options);
fval = -fval;


function f= Corr2_(SimMatCell,AdjMat, useShift,X)
% X: w1, w2,....wn, k1,k2,...,kn
option.useShift =useShift;
sum_mat = CombineSimilarityMatrices(SimMatCell,X,option);
% # % objective function: max ->min
f= - corr2(sum_mat-diag(diag(sum_mat)),AdjMat);

% disp(-f)
