function [ aveBat ] = MLE( likelihoodfile )

data = importdata(likelihoodfile,'\t',1);

atBat0Id = data.data(:,2)==0;
atBat1Id = data.data(:,2)~=0;

aveBat = zeros(size(data.data,1),1);

aveBat(atBat0Id,1) = data.data(atBat0Id,3);
aveBat(atBat1Id,1) = data.data(atBat1Id,3)./data.data(atBat1Id,2);

end
