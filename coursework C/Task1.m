function Task1()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    
    TCGA = csvread("data/TCGA-PANCAN-HiSeq-801x20531/data.csv", 1, 1); % http://archive.ics.uci.edu/ml/machine-learning-databases/00401/
    TCGA = TCGA(:,1:5000); % Take a subset due to processor melting
    
    ds = [1,2,4,8,16,32,64,128,256,512];
    es_pca_df = zeros(10,1);
    es_pca_gram_df = zeros(10,1);
    es_pca_matlab_df = zeros(10,1);
    
    es_pca_tc = zeros(10,1);
    es_pca_gram_tc = zeros(10,1);
    es_pca_matlab_tc = zeros(10,1);
    
    
    dfc = center(DFdata);
    tcc = center(TCGA);
    
    for i = 1:10
        ds(i)
        [~, e_pca_df] = pca_vanilla(DFdata, dfc, ds(i), true);
        [~, e_pca_gram_df] = pca_gram(DFdata, dfc, ds(i), true);
        
        pca_matlab_df = pca(DFdata);
        e_pca_matlab_df = reconstruction_error(DFdata, pca_matlab_df(:,1:ds(i)));
        
        es_pca_df(i) = e_pca_df;
        es_pca_gram_df(i) = e_pca_gram_df;
        es_pca_matlab_df(i) = e_pca_matlab_df;
        
        [~, e_pca_tc] = pca_vanilla(TCGA, tcc, ds(i), true);
        [~, e_pca_gram_tc] = pca_gram(TCGA, tcc, ds(i), true);
        
        pca_matlab_tc = pca(TCGA);
        e_pca_matlab_tc = reconstruction_error(TCGA, pca_matlab_tc(:,1:ds(i)));
        
        es_pca_tc(i) = e_pca_tc;
        es_pca_gram_tc(i) = e_pca_gram_tc;
        es_pca_matlab_tc(i) = e_pca_matlab_tc;
    end
    
    es_pca_df, es_pca_gram_df, es_pca_matlab_df
    es_pca_tc, es_pca_gram_tc, es_pca_matlab_tc
end

function [basis_vecs, e] = pca_vanilla(m, mc, d, compute_error)
    C = cov(mc);
    [V, D] = eig(C);
    [~, ind] = sort(diag(D), 'descend'); % Fetch the top d eigenvalues and indices
    basis_vecs = V(:,ind(1:d)); % Select the top d eigenvectors
    if compute_error
        e = reconstruction_error(m, basis_vecs);
    end
end

function [basis_vecs, e] = pca_gram(m, mc, d, compute_error)
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
    mc = center(m);
    MC_res = mc * (basis_vecs * basis_vecs.');
    error_mat = norm(mc - MC_res, 'fro');
    e = sum(error_mat);
end

function mc = center(m)
    m_mean = mean(m, 1);                            
    mc = m;
    for i=1:size(m,1)
        mc(i,:) = mc(i,:) - m_mean;
    end
end