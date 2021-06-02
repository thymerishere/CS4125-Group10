function Task2()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    [U, e] = snapshot_pca(DFdata, 10, true); % execute with transpose in order to get a matrix of n x m
    e
end

function [U, e] = snapshot_pca(M, d, compute_error)
    center = mean(M, 1);
    Y = M - center;  % Y is the matrix of centered points
    G = Y.' * Y;     % Gram matrix
    Msize = size(M);
    n = Msize(1);
    m = Msize(2);
    [V, D] = eig(G/m);% V is a matrix with columns of eigenvectors and D is a matrix with as diagonal eigenvalues
    U = zeros(n, d);
    for i = 1:d
        U(:,i) = 1/sqrt(D(i,i)) * Y * V(:,i); % calculate basis vectors b
    end
    feat_vec = V(:,1:d);
    if compute_error
         e = reconstruction_error(M, U, feat_vec);
    end
end


function e = reconstruction_error(M, U, feat_vec)
    M_hat = U * feat_vec.';
    error_mat = norm(M - M_hat, 'fro');
    e = sum(error_mat);
end
