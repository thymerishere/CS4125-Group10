function Task4()
    "Loading data"
    FisherIris = load("fisheriris.mat");
    OvarianCancer = load("ovariancancer.mat");
    meas = FisherIris.meas;
    species = FisherIris.species;
    [mnist_data, mnist_labels] = readMNIST("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 10000, 0);
    mnist_data = reshape(mnist_data, [400,10000]);
    "Completed Loading data"
    X = OvarianCancer.obs;
    Y = OvarianCancer.grp;
    calculate_and_plot(X,Y, "Dimension reduction using t-SNE on the Ovarian Cancer dataset", "Dimension reduction using MDS on the Ovarian Cancer dataset.")
    I = randsample(10000, 1000);
    X = mnist_data(:,I).';
    Y = mnist_labels(I,:);
    calculate_and_plot(X,Y, "Dimension reduction using t-SNE on the MNIST dataset", "Dimension reduction using MDS on the MNIST dataset.")
    X = meas;
    Y = species;
    calculate_and_plot(X,Y, "Dimension reduction using t-SNE on the Fisher Iris dataset", "Dimension reduction using MDS on the Fisher Iris dataset.")

end


function calculate_and_plot(dataset, labels, title1, title2)
    figure();
    [Y1, loss1] = tsne(dataset,"Algorithm", "exact", "Distance", "euclidean", "NumDimensions", 2);
    loss1
    gscatter(Y1(:,1),Y1(:,2), labels)
    title(title1)
    "Done with t-SNE"
    figure();
    D = pdist(dataset);
    [Y2,stress] = mdscale(D,2, "Start", "random");
    stress
    gscatter(Y2(:,1),Y2(:,2), labels)
    title(title2)
    "Done with MDS"
end
