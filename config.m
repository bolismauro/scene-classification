% Numero di descrittori sift da utilizzare come codeword
NUM_CODEWORDS = 600;
% Numero di worker da lanciare
NUM_POOL = 4;
% Numero di valori (per canale) utilizzato per determinare cubo utilizzato
% nella quantizzazione
RGB_QUANTIZATION = 8;
% Cubo colore
% La funzione colorcube torna sempre gli stessi colori se il valore passato
% è lo stesso
colorMap = colorcube( RGB_QUANTIZATION * RGB_QUANTIZATION * RGB_QUANTIZATION);