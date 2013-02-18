% File utilizzato per il training del classificatore
clear all
clc

% Aggiungo librerie
addpath('sift');
addpath('svm');
addpath('yael_kmeans');

% carico impostazioni
eval('config');


% carico immagini e sift
% le immagini di training vengono caricate se non è specificato alcun
% parametro
[images names] = loadImages();

% Calcolo i SIFT delle immagini
% Se su disco sono già presenti queste informazioni le carico e non le
% ricalcolo
fprintf('\nCalcolo sift');
if exist('data/sift.mat', 'file') == 2
    fprintf('\ncarico sift da file');
    load 'data/sift'
else
    tic();
    sift_descriptors = cell(1, length(images));
    % richiedo i worker da matlab
    matlabpool(NUM_POOL);
    % for parallelo
    parfor i=1:length(images)
        % calcolo i sift
        [frm,desc] = sift(rgb2gray(images{i}));
        sift_descriptors{i} = desc;
    end
    % chiudo connessione con i worker
    matlabpool close;
    % e li salvo per esecuzioni future
    save('data/sift.mat', 'sift_descriptors');
    fprintf('\nElaborazione sift terminata (%f secondi)',toc());
end


% calcolo codewords
all_desriptors = [];
% unisco tutti i descrittori trovati
for i=1:length(images)
    all_desriptors = [all_desriptors, sift_descriptors{i}];
end
all_desriptors = all_desriptors';

% Anche in questo caso, se ho le codeword su disco utilizzo quelle
% memorizzate anziche ricalcolarle
if exist('data/codewords.mat', 'file') == 2
    load 'data/codewords'
    fprintf('\ncodewords caricate da file');
else
    fprintf('\nAvvio ricerca codewords con kmeans\n');
    tic();
    % MATLAB version, molto lenta
    %[IDX, codewords] = kmeans(all_desriptors, NUM_CODEWORDS, 'Display', 'iter', 'Distance', 'cityblock');
    
    % YAEL KMEANS
    % numero di cluster da creare
    options.K = NUM_CODEWORDS;
    % visualizzo operazioni in corso
    options.verbose = 1;
    % -1 comunica al kmeans di utilizzare tutti i core
    options.num_threads = -1;
    options.max_ite = 100;
    % calcolo codewords (il primo valore tornato sono i centroidi)
    [codewords, dis, assign , nassign , qerr] = yael_kmeans(all_desriptors' , options);
    codewords = codewords';
    fprintf('\nKmeans terminato (%f secondi)',toc());
    save('data/codewords.mat', 'codewords');
end


% calcolo istogrammi
fprintf('\nCalcolo istogrammi');
tic();

% variabile per contenere gli istogrammi. Il numero di beans è il numero di
% codewords più il numero di colori in cui quantizzo l'immagine
histograms = zeros(length(images), NUM_CODEWORDS + size(colorMap,1));
% vettore per contenere le label delle immagini di training
training_label_vector = zeros(length(images), 1);

t_cod = codewords';
% richiedo worker
matlabpool(NUM_POOL);
% for parallelo
parfor i=1:length(images)
    % estraggo i dati dai vettori (sift, immagine)
    sift =  sift_descriptors{i};
    img = images{i};
    % calcolo istrogramma
    histograms(i,:) = getHistogram( sift, NUM_CODEWORDS, t_cod, img, colorMap);
    % estraggo dal nome dell'immagine la label (primo carattere)
    name = names{i};
    training_label_vector(i) = hex2dec(name(1:1)); 
end
matlabpool close;
fprintf('\n calcolo istogrammi terminato terminato (%f secondi)',toc());


fprintf('\nAvvio addestramento SVM');
tic();
% trovo parametri per svm
bestcv = 0;
for log2c = -7:2:15
  for log2g = -12:2:7
    %-v n : n-fold cross validation mode
    cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(training_label_vector, histograms, cmd);
    if (cv >= bestcv),
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end

% addestro svm utilizzando valori trovati
% MultiClass: strategia One-vs-One
cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg)];
svmModel = svmtrain(training_label_vector, histograms, cmd);
fprintf('addestramento SVM terminato (%f secondi)\n',toc());


% persistenza dati (codewords, istogrammi, modello SVM)
save('data/histograms.mat', 'histograms');
save('data/svmModel.mat', 'svmModel');
