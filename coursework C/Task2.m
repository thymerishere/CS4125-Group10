function Task2()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    [U1, e1] = nystrom_method(DFdata.',2 , 10); % execute with transpose in order to get a matrix of n x m
    [U2, e2] = snapshot_pca(DFdata.', 2, 200); % execute with transpose in order to get a matrix of n x m

    e1
    e2
end

function [U, e] = snapshot_pca(M, d, l)
    [n, m] = size(M);
    
    %% Snapshot PCA
    
    % Compute center and centered points
    center = mean(M, 1);
    Y = M - center;  
    
    % Randomly sample l of the m datapoints
    S = sort(randsample(m,l));

    % Compute Gram matrix of sampled datapoints
    subY = Y(:,S);
    G = subY.' * subY;  
    
    % Compute eigenvectors/eigenvalues of Gram matrix (divided by l < m)
    [V, L] = eig(G/l); 
    L = diag(L);  % map diagonal eigenvalues to vector
    % Get max d eigenvalues with their eigenvectors
    [K, I] = maxk(L, d);  % I stores the indices of top eigenvalues
    
    % Compute the basis vectors of the affine spaces
    U = zeros(n, d);
    for i = 1:d
        % Map eigenvectors from R^m to R^n
        U(:,i) = 1/sqrt(L(I(i))) * subY * V(:,I(i)); % calculate basis vectors b
    end
    
    %% Reconstruction error
    % Center vector is replicated to size of data matrix
    C = repmat(center, n, 1).';
    % Map centered data onto affine subspace and map back
    A = (Y.' * U)* U.' + C;
    % Square all errors
    E = (M - A.').^2;
    % Compute error 
    e = sum(E, 'all');
end

function [U, e] = nystrom_method(M, d, l)
    %% Nystrom Method
    [n,m] = size(M);
    % Sample l landmark coordinates into S
    S = randsample(n,l);
    notS = setdiff(1:n, S).';
    % Create permutation matrix P based on S
    p = cat(1,S,notS);
    I = eye(n);
    P = I(p,:);  
    % Pinv is the inverse permutation matrix
    Pinv = P.';  
    % Compute center en calculate centered points
    center = mean(M,1);
    Y = M - center;
    % Permute centered points
    permY = P*Y;  % pre-multiplying to permute the rows
    subY = Y(S,:);  % create l x m matrix 
    % Calculate convariance matrix of n x l
    % TODO: check this new computation
    Cov = (1/m) * permY * subY.'; 
    
    % Separate covariance matrix
    A = Cov(1:l,:);
    B = Cov(l+1:size(Cov,1),:);
    
    % Decompose A into eigenvectors and eigenvalues
    [Ua,La] = eig(A);
    
    % Select top l eigen values and eigen vectors
    [K, I] = maxk(diag(La), l);  % I stores the indices of top eigenvalues
    
    % Reorder eigenvalues and eigenvectors
    La = diag(K);
    Ua = Ua(:,I);
    
    % Create final PCA modes
    Utilde = [Ua; B*Ua*inv(La)]; 
    % Restore original order using inverse permutation matrix
    U = Pinv * Utilde;
    % First d columns of U are the PCA modes
    U = U(:, 1:d);  
    
    %% Reconstruction error
    % Center vector is replicated to size of data matrix
    C = repmat(center, n, 1).';
    % Map centered data onto affine subspace and map back
    A = (Y.' * U)* U.' + C;
    % Square all errors
    E = (M - A.').^2;
    % Compute error 
    e = sum(E, 'all');
end

