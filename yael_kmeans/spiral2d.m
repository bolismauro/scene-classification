function Z = spiral2d(N)

%
% Z = spiral2d(N);
%

t          = rand(1 , N);
temp1      = 2 + 10*t;
temp2      = 7*pi*t;
Z          = [temp1.*sin(temp2) ; temp1.*cos(temp2)] + 0.5*randn(2 , N);

