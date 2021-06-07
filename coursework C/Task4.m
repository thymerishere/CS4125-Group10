function Task4()
    FisherIris = load("fisheriris.mat");
    OvarianCancer = load("ovariancancer.mat");
    meas = FisherIris.meas;
    species = FisherIris.species;
    %calculate_and_plot(meas,species,"Dimension reduction using t-SNE on the Fisher Iris dataset.", "Dimension reduction using Multi Dimensional Scaling on the Fisher Iris dataset.")
    
    X = OvarianCancer.obs;
    Y = OvarianCancer.grp;
    calculate_and_plot(X,Y, "Dimension reduction using t-SNE on the Ovarian Cancer dataset", "Dimension reduction using Multi Dimensional Scaling on the Ovarian Cancer dataset.")
    
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
