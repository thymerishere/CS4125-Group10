function Task1()
    DrivFace = load("data/DrivFace/DrivFace.mat"); % http://archive.ics.uci.edu/ml/datasets/DrivFace
    DFdata = DrivFace.drivFaceD.data;
    [M_res, e] = pca(DFdata, 10, true);
    e
end

function [M_res, e] = pca(m, d, compute_error)
    mc = detrend(m, 'constant');
    mean(mc, 1); % Very close to [0]^N but not quite. 
    C = cov(mc);
    [V, D] = eig(C);
    [B, I] = maxk(maxk(D, 1), d); % Bit useless as the last `d` eigen values are the biggest
    feat_vec = V(:,end-d+1:end);
    M_res = (feat_vec.' * m.').';
    if compute_error
        e = reconstruction_error(m, M_res, feat_vec);
    end
end

function e = reconstruction_error(m, M_res, feat_vec)
    M_hat = M_res * feat_vec.';
    error_mat = norm(m - M_hat, 'fro');
    e = sum(error_mat);
end