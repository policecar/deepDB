
% load initial embeddings ( if exist )
if (params.init_no == 1)
	
	load([params.data_path '/initEmbed.mat']);
    E = We;
	params.num_words = length(words);

% else initialize random embeddings
else
	
	% load vocabulary into a 1 x num_words cell
	fid = fopen([params.data_path '/words.txt'], 'r');
	file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
	fclose(fid);
	file_lines = file_lines{1};
	words = file_lines';
	word_dict = containers.Map(file_lines,1:length(file_lines));
	params.num_words = length(words);

	% generate embeddings
    r = 0.001;
    E = rand(params.embedding_size, params.num_words) * 2 * r - r;
	
	% generate tree with num_entities structs w/ the fields ids, kids, num
	tree = cell(params.num_entities, 1);
	entities = entity_dict.keys;
	for e = 1 : params.num_entities	
		entity = entities(e);
		s_tmp = regexp( entity, '_', 'split'); % TODO: make robust
		ids = [];
		for i = 1 : length(s_tmp)
			word = s_tmp{1}{i};
			if word_dict.isKey(word)
				ids(end+1) = word_dict(word);
			end
		end
		tree{e}.ids = ids;
		tree{e}.kids = [];
		tree{e}.num = length(ids);
    end
	
	% save data structures to file
	We = E;
	save([params.data_path '/initEmbed.mat'], 'We', 'words', 'tree');
	clear We;
	
end

params.tree = tree;
clear tree;

r = 1 / sqrt(params.embedding_size * 2);
W1 = cell(params.num_relations, 1);
W2 = cell(params.num_relations, 1);
b1 = cell(params.num_relations, 1);
U =  cell(params.num_relations, 1);

for i = 1 : params.num_relations
    W1{i} = rand(params.embedding_size, params.embedding_size, params.slice_size) * 2 * r - r;
    if params.inTensorKeepNormal
        W2{i} = rand(params.embedding_size * 2, params.slice_size);
    else
        W2{i} = zeros(params.embedding_size * 2, params.slice_size);
    end
    b1{i} = zeros(1, params.slice_size);
    U{i} = ones(params.slice_size, 1);
end
[params.theta params.decodeInfo] = param2stack(W1, W2, b1, U, E);
