clear;

addpath(genpath('toolbox'));
defaultParams;
output_folder = '../output/';

if (params.actFunc == 0)
    params.actFunc = 'tanh';
    params.f = @(x) (tanh(x));
    params.df = @(z) (1-z.^2);
elseif (params.actFunc == 1)
    params.actFunc = 'sigmoid';
    params.f = @(x) (1./(1 + exp(-x)));
    params.df = @(z) (z .* (1 - z));
elseif (params.actFunc ==2)
    params.actFunc = 'identity';
    params.f = @(x) (x);
    params.df = @(z) (ones(size(z)));
end

if (params.data_no == 0)
    params.data_path = '../data/Wordnet/';
elseif (params.data_no == 1)
    params.data_path = '../data/Freebase/';
end

%% Read from entities.txt
disp('Read EntityFile...');
fid = fopen([params.data_path '/entities.txt'], 'r');
file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
fclose(fid);
file_lines = file_lines{1};

entity_dict = containers.Map(file_lines,1:length(file_lines));
params.num_entities = size(file_lines, 1);
disp('Done.');

%% Read from relations.txt
disp('Read RelationFile...');
fid = fopen([params.data_path '/relations.txt'], 'r');
file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
fclose(fid);
file_lines = file_lines{1};

relation_dict = containers.Map(file_lines,1:length(file_lines));
params.num_relations = size(file_lines, 1);
disp('Done.');

disp('Read TrainFile...');
fid = fopen([params.data_path '/train.txt'], 'r');
file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
fclose(fid);
file_lines = file_lines{1};
params.num_train = size(file_lines, 1);
params.batch_size = min(params.num_train, params.batch_size);
train_data = zeros(params.num_train, 3);
for i = 1 : params.num_train
    s_tmp = regexp(file_lines(i), '\t', 'split');
    train_data(i, 1) = entity_dict(s_tmp{1}{1});
    train_data(i, 2) = relation_dict(s_tmp{1}{2});
    train_data(i, 3) = entity_dict(s_tmp{1}{3});
end
disp('Done.');

load([params.data_path '/initEmbed.mat']);    
params.tree = tree;
params.num_words = length(words);
clear tree; 

disp(params);
initialization;
save([output_folder '0.mat'], 'params');

if strcmp(params.gradient_checking, 'on')
    lst = 1:10;
    data.rel = train_data(lst, 2);
    data.e1 = train_data(lst, 1);
    data.e2 = train_data(lst, 3);
    data.e3 = randi(params.num_entities, length(lst), 1);

    maxDiff = checkNumericalGradient(@(p) tensorNetCostFct(p, data, params, 0), rand(size(params.theta)));
    fprintf('Gradient checking - maxDiff = %.10f\n', maxDiff);
    if (maxDiff > 1e-6)
        fprintf('Failed passing gradient checking....');
        return;
    end

    maxDiff = checkNumericalGradient(@(p) tensorNetCostFct(p, data, params, 1), rand(size(params.theta)));
    fprintf('Gradient checking - maxDiff = %.10f\n', maxDiff);
    if (maxDiff > 1e-6)
        fprintf('Failed passing gradient checking....');
        return;
    end
end

options.Method = 'lbfgs';
options.maxIter = 5;
options.display = 'off';
options.DerivativeCheck = 'off';

fid = fopen([output_folder 'log.txt'], 'w');
tic
for iter = 1 : params.num_iter
    lst = randsample(1: params.num_train, params.batch_size);
    disp(['Iter: ' num2str(iter)]);
    data_batch.rel = repmat(train_data(lst, 2), params.corrupt_size, 1);
    data_batch.e1 = repmat(train_data(lst, 1), params.corrupt_size, 1);
    data_batch.e2 = repmat(train_data(lst, 3), params.corrupt_size, 1);
    data_batch.e3 = randi(params.num_entities, params.batch_size * params.corrupt_size, 1);
    
    if (params.train_both == 1 && rand() < 0.5)
        [params.theta, cost] = minFunc(@(p) tensorNetCostFct(p,data_batch, params, 0),params.theta, options);
    else
        [params.theta, cost] = minFunc(@(p) tensorNetCostFct(p,data_batch, params, 1),params.theta, options);
    end    
    params.cost(iter) = cost;

    fprintf(fid, 'Iter %d: cost = %.6f\n', iter, params.cost(iter));
    display(['cost = ' num2str(params.cost(iter))]);
    
    if (mod(iter, params.save_per_iter) == 0 || iter == params.num_iter)
        save([output_folder num2str(iter) '.mat'], 'params');
        tic
    end
end
fclose(fid);