function [ aveBat ] = MAP( priorfile, likelihoodfile )

likelihood = MLE(likelihoodfile);
prior = MLE(priorfile);

aveBat = likelihood.*prior;

end

