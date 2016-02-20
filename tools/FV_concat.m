function FullFV = FV_concat(numClusters, exemplarSize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FV Concatenating
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run('../tools/vlfeat/toolbox/vl_setup')
[DAT, LB, FNS] = loadAll('..');

%extract the MFCC
mfcc = cell(1,length(DAT));

for i = 1:length(DAT)
    mfcc{i} = DAT{i}.mfc;
end

%create the structure used as input into the demo_fv
GENDATA.data = mfcc;
GENDATA.class = LB;
GENDATA.classnames = {'Blues', 'Classical', 'Country', 'Disco', 'Hiphop',...
	'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'};
%run fisher vector
FV = demo_fv(GENDATA, numClusters, exemplarSize);
filename = ['../generated-fv/FV' int2str(numClusters) '-' int2str(exemplarSize) '.mat']
save(filename,'FV');
save('../generated-fv/LB.mat','LB');
