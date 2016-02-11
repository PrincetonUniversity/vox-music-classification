function FullFV = fv_all_mfc()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FV Concatenating
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run('vlfeat/toolbox/vl_setup')
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
FV = demo_fv(GENDATA, 3, 3);
save('../data/FV.mat','FV');
save('../data/LB.mat','LB');
