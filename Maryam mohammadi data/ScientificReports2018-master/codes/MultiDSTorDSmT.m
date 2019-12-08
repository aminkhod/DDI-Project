function [Label,Belief, ProbV_all]=MultiDSTorDSmT(Proximity,Flag)
% 根据参数选择DS、DSmT或是根据冲突率动态选择DS和DSmT
% <Input>:
%   Proximity   --  一条样本由多个分类器判别形成的mass值,0,A&B,A,B,A|B
%                   [nClassifier X 5]
%   Flag        --  选择开关,0-DS,1-DSmT,2-SwitchByConflictRate,
% <Output>:
%   Label       --  样本融合判别结果,scalar
%   Belief      --  判别Belief值,[5 X 1]
% <See Also>:
%   DST.m  DSmT.m  SwitchByConflictRate.m  BelDST.m    BelDSmT.m
%
% Copyright 2002-2005,信息与控制研究所 reserves all rights.
% Author:施建宇(Shijianyu), snake5947@msn.com
% Created Time 05/13/2005,Last modified:05/13/2005
switch(Flag)
    case 0 
        TMP='DS';
    case 1 
        TMP='DSmT';
    case 2 
        TMP='switch DS or DSmT';
    otherwise 
        error('Error Flag,only 0,1,2 are available.');
end
[nClassifer,nFocalEle]=size(Proximity);
if(nClassifer<2 || nFocalEle~=5)
    error('第一个输入参数不合理，应该多与1行，每行包含5个元素');
end
%------------------------------------------------------------%
e1=Proximity(1,:)';%是一个5维的向量,Proximity{S}(1,:)'包含5个元素
e2=Proximity(2,:)';
switch(Flag)
    case 0 
        [MASS]=DST(e1([1,3:end]),e2([1,3:end]));
    case 1 
        [MASS]=DSmT(e1,e2);
    case 2 
        [MASS,CR]=SwitchByConflictRate(e1,e2);
    otherwise 
        error('Error Flag,only 0,1,2 are available.');
end
 
MASS=NormalizeV(MASS);

for L=3:nClassifer
    ee=Proximity(L,:)';
    switch(Flag)
    case 0 
        [MASS]=DST(MASS,ee([1,3:end]) );
    end
    MASS=NormalizeV(MASS);
end
% 3 根据组合置信度计算置信函数Belief(所有焦元)的值,最大主焦元可以看作是类别
%     Mass=[1;1;(PriorProb').^(-nClassifer+1);1].*Mass;
Belief=zeros(5,1);Label=0;
switch(Flag)
    case 0
        Belief=BelDST(MASS);[ProbV,Label]=max(Belief(2:3)); ProbV_all = Belief(2:3);
end
%     Belief=[1;1;PriorProb';1].*Belief;

function NV=NormalizeV(V)
%归一化向量
% <input>:
% V     --  Vector
% <output>:
% NV    --  Normalized Vector
% Copyright 2002-2005,信息与控制研究所 reserves all rights.
% Author:施建宇(Shijianyu), snake5947@msn.com
% Created Time 04/26/2005,Last modified:04/26/2005
SUM_V=sum(V)*ones(size(V));
NV=V./SUM_V;

%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [M,Q]=DST(e1,e2)
%   calculate Dempster's rule, the format is:
%   [M,Q]=DST(e1,e2)
%   e1,e2: evidences to be combined,they are all vector for a  column.
%   M:     output combined basic belief assignments  for a  column
%   Q:     conflict rate for a number
% Interpretion:
%   The function can only calculate two or three classes for the Frame.If
%   the Frame have two elements, they are denoted in Theta={A,B},and if it
%   have three elements,the Theta={A,B,C}. Then the 2^Theta={0,A,B,AB}
%   for  two dimensions,and the  2^Theta={0,A,B,AB,C,AC,BC,ABC} for three  
%   classes .In the representation,A or B or C is represented by ABC,and
%   others can be understand in this way. Then the Dempster's rule can
%   be calculate in matrix .The evidences can be denoted in 
%   evdence1=[m1(0),m1(A),m1(B),m1(AB)],and
%   evdence1=[m2(0),m2(A),m2(B),m2(AB)] in the real world. m1(AB) is
%   represented by AB1, others can be understand in this way. 

%   Do not hesitate to contract  me ,if you have any question.

[row,col]=size(e1);
%Judge the classes of the Frame
if row==4
     class=2;
     elseif row==8 
             class=3;             
     else disp(['The class number of the Frame is not two or three,the function can not be calculated']);
             return;        
end

if class==2
     A=[0,e1(3),e1(2),0];
     Q=A*e2;
     K=1-Q;
     Transfer=[0,0,0,0;0,e1(2)+e1(4),0,e1(2);0,0,e1(3)+e1(4),e1(3);0,0,0,e1(4)];
     M=Transfer*e2/K;
 end
 if class==3
     A=[0,e1(3)+e1(5)+e1(7),e1(2)+e1(5)+e1(6),e1(5),e1(2)+e1(3)+e1(4),e1(3),e1(2),0];
     Q=A*e2;
     K=1-Q;
     Transfer=[0,0,0,0,0,0,0,0;
               0,e1(2)+e1(4)+e1(6)+e1(8),0,e1(2)+e1(6),0,e1(2)+e1(4),0,e1(2);
               0,0,e1(3)+e1(4)+e1(7)+e1(8),e1(3)+e1(7),0,0,e1(3)+e1(4),e1(3);
               0,0,0,e1(4)+e1(8),0,0,0,e1(4);
               0,0,0,0,e1(5)+e1(6)+e1(7)+e1(8),e1(5)+e1(7),e1(5)+e1(6),e1(5);
               0,0,0,0,0,e1(6)+e1(8),0,e1(6);
               0,0,0,0,0,0,e1(7)+e1(8),e1(7);
               0,0,0,0,0,0,0,e1(8)];
      M=Transfer*e2/K;
 end
     
 
 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Bel=BelDST(mass)
%   calculate the Belief Function from bba m(.) in DST, the format is:
%   Bel=BelDST(mass)
%   mass: mass function,it is a vector for a  column.It have 4 elements
%        for two  class and 8 elements for three class ,the size is n*1.
%   Bel: output the Belief Function from gbba m(.),it have 4 elements
%        for two  class and 8 elements for three class ,the size is n*1.
% Interpretion:
%   The function can only calculate two or three classes for the Frame
%   within DSmT.If the Frame have 4 elements,  they are denoted in
%   Theta={A,B},and if it  have 8 elements,the Theta={A,B,C}. Then the
%   2^Theta={0,A,B,A|B} for  two dimensions ,2^Theta={0,A,B,A|B,C,A|C,B|C,A|B|C}
%   for three dimesions.  Then the Belief Function   from bba m(.) in DST
%   can be calculate in matrix .    

%   Do not hesitate to contract  me ,if you have any question.

[row,col]=size(mass);
%Judge the classes of the Frame
if row==4
    class=2;
    elseif row==8
            class=3;
    else disp(['The class number of the Frame is not two or three,the function can not be calculated']);
        return;    
end

if class==2
    Bel=[1,0,0,0;
         1,1,0,0;
         1,0,1,0;
         1,1,1,1;]*mass;
end
if class==3
    Bel=[1,0,0,0,0,0,0,0;
         1,1,0,0,0,0,0,0;
         1,0,1,0,0,0,0,0;
         1,1,1,1,0,0,0,0;
         1,0,0,0,1,0,0,0;
         1,1,0,0,1,1,0,0;
         1,0,1,0,1,0,1,0;
         1,1,1,1,1,1,1,1;            
         ]*mass;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

