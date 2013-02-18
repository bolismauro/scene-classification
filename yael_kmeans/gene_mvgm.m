function [M , S , P] = gene_mvgm(d , K , sig)

%
%
% [M , S , P] = gene_mvgm(d , K , sig);
%
%
%


eyed   = eye(d);


S      = zeros(d , d , K);


P      = rand(1 , 1 , K);
sP     = sum(P , 3);
P      = P./sP(: , : , ones(1  , K));

M      = 1.25*randn(d , 1 , K);


for k=1:K
      
    % Compute variance matrix
    
    Rot = eyed;
    for dd=1:d
        for e=(dd+1):d
            theta      = rand(1) * 2*pi;
            rot        = eyed;
            rot(dd,dd) = cos(theta);
            rot(e,e)   = rot(d,d);
            rot(dd,e)  = sin(theta);
            rot(e,dd)  = -rot(d,e);
            Rot        = rot*Rot;
        end
    end

    Rot          = sig*Rot*diag(randn(d,1) + 1);
    S(: , : , k) = Rot*Rot';
   
end

