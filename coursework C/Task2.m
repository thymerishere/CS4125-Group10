function Task2()
    %% Plots of data transformed using PCA modes


    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    labels = DrivFace.drivFaceD.nlab; 
    M = DFdata;
    Y = labels;
 
    size(M)
    
    [U1, e1] = nystrom_method(M.', 2, 500); % execute with transpose in order to get a matrix of n x m
    [U2, e2] = snapshot_pca(M.', 2, 100); % execute with transpose in order to get a matrix of n x m
    D1 = (M - mean(M,2)) * U1;
    D2 = (M - mean(M,2)) * U2;
    figure
    gscatter(D1(:,1),D1(:,2), Y)
    figure
    gscatter(D2(:,1),D2(:,2), Y)
    
    %% Graphs plotting experiment of Time/RE tradeoff for L
    L1 = [605, 500, 400, 300, 200, 100, 50, 25, 10];
    L2 = [6399, 5000, 4000, 3000, 2000, 1000, 500, 250, 100, 50];

    [T1,E1,T2,E2] = experiment_data(L1, L2);
    figure
    yyaxis left
    errorbar(L1,mean(T1,2),std(T1,0,2))
    title("Execution Time and RE over varying values of L running Snapshot PCA")
    hold on
    yyaxis right
    errorbar(L1,mean(E1,2),std(E1,0,2))
    legend('Time in seconds', 'Reconstruction Error', 'Location','northwest')
    hold off
    figure
    yyaxis left
    errorbar(L2,mean(T2,2),std(T2,0,2))
    title("Execution Time and RE over varying values of L running Nystrom")
    hold on
    yyaxis right
    errorbar(L2,mean(E2,2),std(E2,0,2))
    legend('Time in seconds', 'Reconstruction Error', 'Location','northwest')
    hold off
    
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
    
    % Compute the basis vectors of the affine subspace
    U = zeros(n, d);
    for i = 1:d
        % Map eigenvectors from R^m to R^n
        U(:,i) = 1/sqrt(L(I(i))) * subY * V(:,I(i)); % calculate basis vectors b
%         U(:,i) = norm(U(:,i));
    end
    
    %% Reconstruction error
    % Map centered data onto affine subspace and map back
    A = Y.' * (U * U.');
    % Square all errors
    E = (Y - A.').^2;
    % Compute error 
    e = sqrt(sum(E, 'all'));
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
    % Map centered data onto affine subspace and map back
    A = Y.' * (U * U.');
    % Square all errors
    E = (Y - A.').^2;
    % Compute error 
    e = sqrt(sum(E, 'all'));
end


function [T1, E1, T2, E2] = experiment_data(L1,L2)
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;

    %L1 = [605, 500, 400, 300, 200, 100, 50, 25, 10];
    %L2 = [6399, 5000, 4000, 3000, 2000, 1000, 500, 250, 100, 50];
    
    d = 2;
    
    T1 = [];
    E1 = [];
    for i = 1:size(L1,2)
        l = L1(i);
        for j = 1:50
            tic 
            [U, E1(i,j)] = snapshot_pca(DFdata.', d, l);
            T1(i,j) = toc;
        end 
    end 
    "Executed Snapshot PCA"
    
    T2 = [];
    E2 = [];
    for i = 1:size(L2,2)
        l = L2(i);
        for j = 1:5
            tic 
            [U, E2(i,j)] = nystrom_method(DFdata.', d, l);
            T2(i,j) = toc;
        end 
    end 
    "Executed Nystrom Method"
    
end
