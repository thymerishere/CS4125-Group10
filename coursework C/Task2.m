function Task2()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    %[U, e] = nystrom_method(DFdata.', 10, true); % execute with transpose in order to get a matrix of n x m
    [U, e] = snapshot_pca(DFdata.', 10, true); % execute with transpose in order to get a matrix of n x m

    e
end

function [U, e] = snapshot_pca(M, d, compute_error)
    center = mean(M, 1);
    Y = M - center;  % Y is the matrix of centered points
    [n, m] = size(M);
    
    l = round(m / 10);  % sample 10 percent of the points
    S = sort(randsample(m,l));

    subY = Y(:,S);
    G = subY.' * subY;     % Gram matrix

    [V, D] = eig(G/l);  % V is a matrix with columns of eigenvectors and D is a matrix with as diagonal eigenvalues
    U = zeros(n, d);
    for i = 1:d
        U(:,i) = 1/sqrt(D(i,i)) * subY * V(:,i); % calculate basis vectors b
    end
    feat_vec = V(:,1:d);
    if compute_error
        % TODO: reconstruction error
         %e = reconstruction_error(M, U, feat_vec);
    end
    
    e = 0;
end

function [U, e] = nystrom_method(M, d, compute_error)
    [n,m] = size(M);
    % Set l as halfway between d and n
    l = (d+n)/2;
    % Select subset s
    S = randsample(n,l);
    
    % TODO: permutations (slide 3)?
    notS = setdiff(1:n, S).';
    p = cat(1,S,notS);
    I = eye(n);
    P = I(p,:);  % P is the permutation matrix
    Pinv = P.';  % Pinv is the inverse permutation matrix
    % Compute center en calculate centered points
    center = mean(M,1);
    Y = M - center;
    
    permY = P*Y;  % pre-multiplying to permute the rows
    subY = Y(S,:);  % create l x m matrix 
    % oldCov = cov(newY);
    
    % TODO: check this new computation
    Cov = (1/m) * permY * subY.'; 
    
    % Separate covariance matrix
    A = Cov(1:l,:);
    B = Cov(l+1:size(Cov,1),:);
    % Decompose A into eigenvectors and eigenvalues
    [Ua,La] = eig(A);
    % Create final PCA modes
    Utilde = [Ua; B*Ua*inv(La)];
    % restore original order
    U = Pinv * Utilde;
    
    % TODO: reconstruction error????
    e = 0
end


function e = reconstruction_error(M, U, feat_vec)
    M_hat = U * feat_vec.';
    error_mat = norm(M - M_hat, 'fro');
    e = sum(error_mat);
end
