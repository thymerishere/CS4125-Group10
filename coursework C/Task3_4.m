fopen('output/task3_4.txt','w');

% Loading data
OvarianCancer = load("ovariancancer.mat");

% Calculating and plotting
X = OvarianCancer.obs;
Y = OvarianCancer.grp;
calculate_and_plot(X,Y, "Ovarian Cancer");

% Loading data
FisherIris = load("fisheriris.mat");

% Calculating and plotting
X = FisherIris.meas;
Y = FisherIris.species;
calculate_and_plot(X,Y, "Fisher Iris");

% Loading data
[mnist_data, mnist_labels] = readMNIST("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 10000, 0);
mnist_data = reshape(mnist_data, [400,10000]);

% Calculating and plotting
I = randsample(10000, 1000);
X = mnist_data(:,I).';
Y = mnist_labels(I,:);
calculate_and_plot(X,Y, "MNIST");

function calculate_and_plot(dataset, labels, datasetname)
    fid = fopen('output/task3_4.txt','a');
    
    % t-SNE
    [Y1, loss1] = tsne(dataset,"Algorithm", "exact", "Distance", "cosine", "NumDimensions", 2);
    gscatter(Y1(:,1),Y1(:,2), labels)
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    saveas(gcf, "figures/task3_4/t-SNE cosine " + datasetname)
    fprintf(fid, "TSNE loss cosine " + datasetname + " " + loss1 + "\n");
    
     % t-SNE
    [Y1, loss1] = tsne(dataset,"Algorithm", "exact", "Distance", "spearman", "NumDimensions", 2);
    gscatter(Y1(:,1),Y1(:,2), labels)
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    saveas(gcf,'figures/task3_4/t-SNE spearman ' + datasetname);
    fprintf(fid, "TSNE loss spearman " + datasetname + " " + loss1 + "\n");
    
    % MDS cosine
    D = pdist(dataset, 'cosine');
    D = squareform(D);
    [Y2,stress,lambda_p] = cmds(D,2);
    gscatter(Y2(:,1),Y2(:,2), labels)
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    saveas(gcf, "figures/task3_4/MDS cosine " + datasetname);
    fprintf(fid, "MDS stress cosine " + datasetname + " " + sum(stress(:)) + "\n");
    fprintf(fid, "MDS lambda 1 cosine " + datasetname + " " + lambda_p(1,1) + "\n");
    fprintf(fid, "MDS lambda 2 cosine " + datasetname + " " + lambda_p(2,2) + "\n");
   
    % MDS spearman
    D = pdist(dataset, 'spearman');
    D = squareform(D);
    [Y2,stress,lambda_p] = cmds(D,2);
    gscatter(Y2(:,1),Y2(:,2), labels);
    xlabel('Dimension 1');
    ylabel('Dimension 2');
    saveas(gcf, "figures/task3_4/MDS spearman "  + datasetname);
    fprintf(fid, "MDS stress spearman " + datasetname + " " + sum(stress(:)) + "\n");
    fprintf(fid, "MDS lambda 1 spearman " + datasetname + " " + lambda_p(1,1) + "\n");
    fprintf(fid, "MDS lambda 2 spearman " + datasetname + " " + lambda_p(2,2) + "\n");
    
    fclose(fid);
    
end


function [Y, stress, lambda_p] = cmds(D, p)
    [n,m] = size(D);
    
    % Compute Gram matrix G using double centering
    In = eye(n); % N-by-n dentity matrix
    e = ones(n, 1); 
    part = In - (1/n) * (e * e.');
    G = -0.5 * part * D * part;
    
    % Compute eigenvalues (lambda) and vectors (u) of G Q: left or right?
    [u,lambda] = eig(G); 
    
    % Obtaining p highest eigenvalues and corresponding vectors
    [lambda_p, index_p] = maxk(diag(lambda), p);
    lambda_p = diag(lambda_p).';
    u_p = u(:,index_p);
    
    % For any d compute the coordinates of the points x_i
    Y = (u_p*sqrt(lambda_p));
    
    % Compute stress
    distances = squareform(pdist(Y));
    desired_distances = D;
    stress = (distances-desired_distances).^2; 
    
end
