function [ images, names ] = loadImages( type )
%IMAGESLOADER Carica in una struttura le immagini in una data cartella.
%Torna un vettore cell
    
    % carico by-default le immagini di training
    if nargin < 1
        db_path = 'db/train';
    else
        db_path = strcat('db/', type);
    end
    
    db = dir(strcat(db_path, '/*.jpg'));
    images = cell(1, length(db));
    names = cell(1, length(db));
    
    for i = 1:length(db)
      img_path = strcat(db_path, '/', db(i).name);
      images{i} = imread(img_path);
      names{i} = db(i).name;
    end
    
    fprintf('Caricate %d immagini', length(db));
    
end

