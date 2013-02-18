% Fast mex K-means clustering algorithm with possibility of K-mean++ initialization
% (mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)
% 
% - Accept single/double precision input
% - Support of BLAS/OpenMP for multi-core computation
% 
% Usage
% ------
% 
% [centroids, dis, assign , nassign , qerr] = yael_kmeans(X , [options]);
% 
% 
% Inputs
% -------
% 
% X                                        Input data matrix (d x N) in single/double format 
% 
% options
%        K                                 Number of centroid  (default K = 10)
%        max_ite                           Number of iteration (default max_ite = 50)
%        redo                              Number of time to restart K-means (default redo = 1)
%        verbose                           Verbose level = {0,1,2,3} (default verbose = 0)
%        init_random_mode                  0 <=> Kmeans++ initialization, 1<=> random selection ...
%        normalize_sophisticated_mode      0/1 (No/Yes)
%        BLOCK_N1                          Cache size block (default BLOCK_N1 = 1024)
%        BLOCK_N2                          Cache size block (default BLOCK_N2 = 1024)
%        seed                              Seed number for internal random generator (default random seed according to time)
% 
% If compiled with the "OMP" compilation flag
% 
%        num_threads                       Number of threads   (default num_threads = max number of core)
% 
% 
% Outputs
% -------
% 
% centroids                               Centroids matrix (d x K) in single/double format 
% dis                                     Distance of each xi to the closest centroid (1 x N) in single/double format
% assign                                  Index of closest centroid to xi, i = 1,...,N in UINT32 format
% nassign                                 Number of data associated with each cluster (1 x K) in UINT32 format
% qerr                                    Quantification error during training process (1 x options.max_ite) in single/double format


%% Example 1: Cluster a mixture of 3 Gaussians mixture %%


clear,close all
K                                   = 3;
mu                                  = cat(3 , [-2.5 ; -3] , [0 ; 0] ,[ 5 ; 5]);                %(d x 1 x M)
sigma                               = cat(3 , [2 0; 0 1] , [2 -.2; -.2 2] , [1 .9; .9 1]  ); %(d x d x M)
p                                   = cat(3 , [0.3] , [0.2]  , [0.5]);                       %(1 x 1 x M)
N                                   = 10000;
[Z , index]                         = sample_mvgm(N , mu , sigma , p);
[x , y]                             = ndellipse(mu , sigma);

options.K                           = 3;
options.max_ite                     = 50;
options.num_threads                 = 2;

tic,[centroids, dis, assign , nassign , qerr]         = yael_kmeans(Z  , options);,toc

figure(1)
plot(Z(1 , :) , Z(2 , :) , 'k+'  , x , y , 'g' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(reshape(mu(1 , : , :) , 1 , K) , reshape(mu(2 , : , :) , 1 , K)  , 'r+'  ,  centroids(1 , :) , centroids(2 , :) , 'mo', 'markersize' , 6);
h = voronoi(double(centroids(1 , :)) , double(centroids(2 , :)) );
set(h ,  'linewidth' , 2)
h = title(sprintf('Kmeans clustering for a %d gaussians mixture pdf, assuming K = %d clusters' , K , options.K));
set(h ,  'fontsize' , 12)

hold off

figure(2)
h = semilogy(1:options.max_ite , qerr);
ylabel('Quantification Error', 'fontsize' , 11)
xlabel('Kmeans iteration' , 'fontsize' , 11)
set(h , 'linewidth' , 2)
h = title(sprintf('Kmeans clustering for a %d gaussians mixture pdf, assuming K = %d clusters' , K , options.K));
set(h ,  'fontsize' , 12)

grid on

%% Example 2: Cluster a mixture of K Gaussians mixture %%

d                                       = 2;
K                                       = 10;
clust_spread                            = 0.1;
N                                       = 20000;



[mu , sigma , p]                        = gene_mvgm(d , K , clust_spread);

[Z , index]                             = sample_mvgm(N , mu , sigma , p);
[x , y]                                 = ndellipse(mu , sigma);

options.K                               = K;
options.init_random_mode                = 0;
options.normalize_sophisticated_mode    = 0;
options.BLOCK_N1                        = 1024;
options.BLOCK_N2                        = 1024;
options.seed                            = 1234543;
options.num_threads                     = -1;


tic,[centroids, dis, assign , nassign , qerr]  = yael_kmeans(single(Z)  , options);,toc

figure(3)
plot(Z(1 , :) , Z(2 , :) , 'k+'  , x , y , 'g' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(reshape(mu(1 , : , :) , 1 , K) , reshape(mu(2 , : , :) , 1 , K)  , 'r+'  ,  centroids(1 , :) , centroids(2 , :) , 'mo', 'markersize' , 6);
h = voronoi(double(centroids(1 , :)) , double(centroids(2 , :)) );
set(h ,  'linewidth' , 2);
h = title(sprintf('Kmeans clustering for a %d gaussians mixture pdf, assuming K = %d clusters' , K , options.K));
set(h ,  'fontsize' , 12);

hold off

figure(4)
h = semilogy(1:options.max_ite , qerr);
ylabel('Quantification Error', 'fontsize' , 11)
xlabel('Kmeans iteration' , 'fontsize' , 11)
set(h , 'linewidth' , 2)
h = title(sprintf('Kmeans clustering for a %d gaussians mixture pdf, assuming K = %d clusters' , K , options.K));
set(h ,  'fontsize' , 12)
grid on


%% Example 3: Cluster uniform random data %%

d                                       = 2;
N                                       = 2000;
Z                                       = rand(d , N);

options.K                               = 50;
options.init_random_mode                = 0;
options.normalize_sophisticated_mode    = 0;
options.BLOCK_N1                        = 1024;
options.BLOCK_N2                        = 1024;
options.seed                            = 1234543;
options.num_threads                     = -1;


tic,[centroids, dis, assign , nassign , qerr]  = yael_kmeans(single(Z)  , options);,toc

figure(5)
plot(Z(1 , :) , Z(2 , :) , 'k+' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(centroids(1 , :) , centroids(2 , :) , 'mo', 'markersize' , 6);
h = voronoi(double(centroids(1 , :)) , double(centroids(2 , :)) );
set(h ,  'linewidth' , 2);
h = title(sprintf('Kmeans clustering for uniform data, assuming K = %d clusters' , options.K));
set(h ,  'fontsize' , 12);

hold off



figure(6)

bar(1:options.K , nassign)
ylabel('Assignement', 'fontsize' , 11)
xlabel('Cluster #' , 'fontsize' , 11)
axis([-0.0 , options.K+0.5 , 0 , 1.1*max(nassign)])
h = title(sprintf('Assignement for the K=%d clusters' ,options.K));


%% Example 4: Cluster spiral data %%

d                                       = 2;
N                                       = 5000;
Z                                       = spiral2d(N);


options.K                               = 50;
options.init_random_mode                = 0;
options.normalize_sophisticated_mode    = 0;
options.BLOCK_N1                        = 1024;
options.BLOCK_N2                        = 1024;
options.seed                            = 1234543;
options.num_threads                     = 0;


tic,[centroids, dis, assign , nassign , qerr]  = yael_kmeans(single(Z)  , options);,toc

figure(7)
plot(Z(1 , :) , Z(2 , :) , 'k+' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(centroids(1 , :) , centroids(2 , :) , 'mo', 'markersize' , 6);
h = voronoi(double(centroids(1 , :)) , double(centroids(2 , :)) );
set(h ,  'linewidth' , 2);
h = title(sprintf('Kmeans clustering for uniform data, assuming K = %d clusters' , options.K));
set(h ,  'fontsize' , 12);

hold off


