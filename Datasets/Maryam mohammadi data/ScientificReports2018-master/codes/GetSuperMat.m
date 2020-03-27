function SuperADJ= GetSuperMat(ADJ,Sim_col, cutoff)
% T: clustering results containing cluster labels of columns
% ADJ - interaction matrix
% Sim_col- similarity matrix between columns
ADJ(ADJ ~=1)= 0;

% # --- clustering
dissimilarity = 1- Sim_col;
Y=squareform(dissimilarity-diag(diag(dissimilarity)) );
Z=linkage(Y,'ward');
if nargin <3
    cutoff = 0.7*(max(Z(:,3)));
end

T = cluster(Z,'cutoff',cutoff,'criterion','distance');

figure,[H,~] = dendrogram(Z,size(Sim_col,1),'colorthreshold',cutoff);'default';
set(H,'LineWidth',2)

% # --- Shrinking
ADJ_shrink = ADJ;

uT=unique(T);
MergedColIdx  =[];
t=0;
for u=1:length(uT)
    idx = find(T==uT(u));
    if length(idx)<1 %
        continue;
    else
        t=t+1;
        MergedColIdx{t,1} = idx;
    end
end

SuperADJ = Merge_(ADJ_shrink,MergedColIdx);

function SuperADJ = Merge_(ADJ_shrink,MergedColIdx)
MergedCol = zeros(size(ADJ_shrink,1),length(MergedColIdx));
MergedColCell = cell(1,length(MergedColIdx));

for t=1:length(MergedColIdx)
    colIdx = MergedColIdx{t};
    for c =1: length(colIdx)
        MergedCol(:,t) =  MergedCol(:,t) | ADJ_shrink(:, colIdx(c) );
    end
    MergedColCell{t}= repmat(MergedCol(:,t),1,length(colIdx));
end

allMergedCol = cell2mat( MergedColIdx );

%
ADJ_shrink(:,allMergedCol) = cell2mat( MergedColCell);

SuperADJ = ADJ_shrink;