function [ histogram ] = getHistogram( sift, NUM_CODEWORDS, codewords, img, colorMap)
    % Funzione che calcola l'istogramma delle codewords e colore
    % dell'immagine utilizzando i parametri passati, i due istogrammi
    % vengono tornati uno di seguito all'altro
    
    [M N] = size(sift);
    %calcolo distanza da ogni punto a ogni centroide
    distance = Inf * ones(N,NUM_CODEWORDS);
    for p = 1:N
      for c = 1:NUM_CODEWORDS
        %distanza L2  
        distance(p,c) = norm(codewords(:,c) - double(sift(:,p)));
      end
    end
    
    % trovo i valori minimi di distanza
    % Specifico 2 per indicare che bisogna confrontare le distanze, per
    % ogni rigacl rispetto alle colonne.
    % ogni riga contiene le distanze di una data codeword.
    % indices contiene gli indici (colonna) dove posso trovare la distanza minima
    [tmp,indices] = min(distance,[],2);

    sift_histogram = zeros(1,NUM_CODEWORDS);
    
    % creo istogramam
    for p = 1:N
      sift_histogram(indices(p)) = sift_histogram(indices(p)) + 1;
    end    
    
    % normalizzo, avrò somma 1
    sift_histogram = sift_histogram ./ N;
    
    
    % creo istogramma colore
    
    % avrò un bean per ogni colore 
    color_histogram = zeros(1, size(colorMap,1));
    % creo mappa di indici a partire dall'immagine
    % "+1" : fix per permettere di usare l'operazione nel for
    % 'nodither' : maps each color in the original image to the 
    %  closest color in the new map. No dithering is performed.
    color_indices = rgb2ind(img, colorMap, 'nodither') + 1; 
    
    [M N Z] = size(img);
    
    % creo istogramma
    for i=1:M
        for j=1:N
            color_histogram(color_indices(i,j)) = color_histogram(color_indices(i,j))+1;
        end
    end
    
    % normalizzo per il numero di colori
    color_histogram = color_histogram./(M*N);
    
    % unisco due istrogrammi
    histogram = [sift_histogram, color_histogram];
    
end



% VERSIONE CON DIVISIONE SPAZIALE COLORE
% function [ histogram ] = getHistogram( sift, NUM_CODEWORDS, codewords, img, colorMap)
%     % Funzione che calcola l'istogramma delle codewords e colore
%     % dell'immagine utilizzando i parametri passati, i due istogrammi
%     % vengono tornati uno di seguito all'altro
%     
%     [M N] = size(sift);
%     %calcolo distanza da ogni punto a ogni centroide
%     distance = Inf * ones(N,NUM_CODEWORDS);
%     for p = 1:N
%       for c = 1:NUM_CODEWORDS
%         %distanza L2  
%         distance(p,c) = norm(codewords(:,c) - double(sift(:,p)));
%       end
%     end
%     
%     % trovo i valori minimi di distanza
%     % Specifico 2 per indicare che bisogna confrontare le distanze, per
%     % ogni rigacl rispetto alle colonne.
%     % ogni riga contiene le distanze di una data codeword.
%     % indices contiene gli indici (colonna) dove posso trovare la distanza minima
%     [tmp,indices] = min(distance,[],2);
% 
%     sift_histogram = zeros(1,NUM_CODEWORDS);
%     
%     % creo istogramam
%     for p = 1:N
%       sift_histogram(indices(p)) = sift_histogram(indices(p)) + 1;
%     end    
%     
%     % normalizzo, avrò somma 1
%     sift_histogram = sift_histogram ./ N;
%     
%     histogram = [sift_histogram];
%     
%     % creo istogramma colore
%     [R C Z] = size(img);
%     COLOR_SQUARE = 4;
%     r_step = fix( R/COLOR_SQUARE );
%     c_step = fix( C/COLOR_SQUARE );
% 
%     
%     % divido l'immagine in sottoblocchi
%     for i=1:COLOR_SQUARE
%         for j=1:COLOR_SQUARE
%                 % avrò un bean per ogni colore 
%                 color_histogram = zeros(1, size(colorMap,1));
%                 
%                 x1 = c_step * (j-1) + 1;
%                 y1 = r_step * (i-1) + 1;
% 
%                 x2 = min(x1 + c_step - 1, C);
%                 y2 = min(y1 + r_step - 1, R);
% 
%                 % estraggo sottoblocco
%                 block = img(y1:y2,x1:x2,:);                
%                 
%                 % creo mappa di indici a partire dall'immagine
%                 % "+1" : fix per permettere di usare l'operazione nel for
%                 % 'nodither' : maps each color in the original image to the 
%                 %  closest color in the new map. No dithering is performed.
%                 color_indices = rgb2ind(block, colorMap, 'nodither') + 1; 
% 
%                 [M N Z] = size(block);
% 
%                 % creo istogramma
%                 for i=1:M
%                     for j=1:N
%                         color_histogram(color_indices(i,j)) = color_histogram(color_indices(i,j))+1;
%                     end
%                 end
% 
%                 % normalizzo per il numero di colori
%                 color_histogram = color_histogram./(R*C);
% 
%                 % unisco due istrogrammi
%                 histogram = [histogram, color_histogram];
%         end
%     end
% 
%     
% end
% 
% 
