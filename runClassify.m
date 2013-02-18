% File di run per la classificazione delle immagini di test
clear all
clc

%carico librerie
addpath('sift');
addpath('svm');

% carico impostazioni
eval('config');

% carico dati creati dal training (modello svm e codewords)
load data/svmModel
load data/codewords

%carico immagini e sift
%specifico di caricare le immagini nella cartella test
[images, names] = loadImages('test');


tic();
sift_descriptors = cell(1, length(images));
num_sift = 0;
% richiedo worker a matlab
matlabpool(NUM_POOL);
% for parallelo
parfor i=1:length(images)
    % estraggo sift
    [frm,desc] = sift(rgb2gray(images{i}));
    sift_descriptors{i} = desc;
end
matlabpool close
fprintf('\nElaborazione sift terminata (%f secondi)',toc());


fprintf('\nCalcolo istogrammi');
% variabile per contenere gli istogrammi. Il numero di beans è il numero di
% codewords più il numero di colori in cui quantizzo l'immagine
histograms = zeros(length(images), NUM_CODEWORDS + size(colorMap,1));
% vettore per contenere le vere label delle immagini
real_labels = zeros(length(images), 1);

t_cod = codewords';
% richiedo worker
matlabpool(NUM_POOL);
% for parallelo, cicla sulle immagini di test
parfor i=1:length(images)
    % estraggo info necessarie
    sift =  sift_descriptors{i};
    img = images{i};
    % calcolo istrogramma
    histograms(i,:) = getHistogram( sift, NUM_CODEWORDS, t_cod, img, colorMap);
    % memorizzo vera label dell'immagine (primo carattere)
    name = names{i};
    real_labels(i) = hex2dec(name(1:1));
end
matlabpool close

fprintf('\n calcolo istogrammi terminato terminato (%f secondi)',toc());

% vettore fittizzio di label (non ho interesse a far calcolare a svmtrain
% l'accuratezza ecc)
fake_labels = zeros(length(images),1);
% richiedo previsioni sugli istogrammi
[predicted_label, accuracy, decision_values] = svmpredict(fake_labels, histograms, svmModel, '-q');

% stampo confusion table
groupVec = ind2vec(real_labels'+1);
labelVec = ind2vec(predicted_label'+1);
plotconfusion(groupVec,labelVec);


%SLIDES: stampo immagini con miss-classification
% for i=1:length(images)
%     if real_labels(i) == predicted_label(i)
%     else
%         fprintf('%s classificata come %d invece che %d\n', names{i}, predicted_label(i), real_labels(i));
%     end
% end
