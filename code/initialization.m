if (params.init_no == 1)
    E = We;
else
    r = 0.001;
    E = rand(params.embedding_size, params.num_words) * 2 * r - r;
end

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
