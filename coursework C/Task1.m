function Task1()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    
    ds = [1,2,4,8,16,32,64,128,256,512];
    es_pca = zeros(10,1);
    es_pca_gram = zeros(10,1);
    es_pca_matlab = zeros(10,1);
    
    for i = 1:10
        ds(i)
        [~, e_pca] = pca_vanilla(DFdata, ds(i), true);
        [~, e_pca_gram] = pca_gram(DFdata, ds(i), true);
        
        pca_matlab = pca(DFdata);
        e_pca_matlab = reconstruction_error(DFdata, pca_matlab(:,1:ds(i)));
        
        es_pca(i) = e_pca;
        es_pca_gram(i) = e_pca_gram;
        es_pca_matlab(i) = e_pca_matlab;
    end
    
    es_pca
    es_pca_gram
    es_pca_matlab
end

function [basis_vecs, e] = pca_vanilla(m, d, compute_error)
    m_mean = mean(m, 1);
    mc = m;
    for i=1:size(m,1)
        mc(i,:) = mc(i,:) - m_mean;
    end
    C = cov(mc);
    [V, D] = eig(C);
    [~, ind] = sort(diag(D), 'descend'); % Fetch the top d eigenvalues and indices
    basis_vecs = V(:,ind(1:d)); % Select the top d eigenvectors
    if compute_error
        e = reconstruction_error(m, basis_vecs);
    end
end

function [basis_vecs, e] = pca_gram(m, d, compute_error)
    m_mean = mean(m, 1);                            % Trivial way to compute mean
    mc = m;
    for i=1:size(m,1)
        mc(i,:) = mc(i,:) - m_mean;
    end
    [V, D] = eig(1/(size(mc, 1))*(mc*mc.'));        % Calculate eigenvalues of (1/m)YY^T (we have row data)
    [D_sorted, ind] = sort(diag(D), 'descend');     % Fetch the top d eigenvalues and indices
    eigen_vecs = V(:,ind(1:d));                     % Select the d best eigenvectors
    basis_vecs = zeros(size(mc, 2), d);             % Pre-allocate
    for i = 1:d
        eigenvalue = D_sorted(i);
        basis_vecs(:,i) = (1/(eigenvalue^0.5)) * mc.' * eigen_vecs(:,i); % Calculate basis vector i
    end
    if compute_error
        e = reconstruction_error(m, basis_vecs);
    end
end

function e = reconstruction_error(m, basis_vecs)
    m_mean = mean(m, 1);
    mc = m;
    for i=1:size(m,1)
        mc(i,:) = mc(i,:) - m_mean;
    end
    MC_res = mc * (basis_vecs * basis_vecs.');
    error_mat = norm(mc - MC_res, 'fro');
    e = sum(error_mat);
end