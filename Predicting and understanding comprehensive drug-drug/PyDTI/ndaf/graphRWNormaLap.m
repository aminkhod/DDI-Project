function A=graphRWNormaLap(A)
n=size(A,1);
A(1:n+1:end)=0;
A=bsxfun(@rdivide,(speye(n)-A),(sum(A,2)));