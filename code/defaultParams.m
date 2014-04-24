if ~exist('params','var')
    params.data_no = 2; % 0 - WordNet, 1 - FreeBase, 2 - Weltmodell
    params.init_no = 0; % 0 - random, 1 - Turian et al
    params.num_iter = 500; 
    params.train_both = 0;
	
	% number of unique words ( whereas num_entities may contain compounds )
	params.num_words = 17905;

    params.batch_size = 20000;
    params.corrupt_size = 10;
    params.embedding_size = 100;
    params.slice_size = 3; 
    
    params.reg_parameter = 0.0001;
    params.actFunc = 0; % 0 - tanh, 1 - sigmoid, 2 -identity
    params.inTensorKeepNormal = 0; 
    
    params.save_per_iter = 100;
    params.gradient_checking = 'on';
end
