function RankMat_ = GetRankMatAlongRow(Mat)
% To get  rank matrix by  matrix,
% SimMat - input matrix
[~,IX_]=sort(Mat,1,'descend');
[~,RankMat_]=sort(IX_,1,'ascend'); % by col, not by row
