
    addpath(genpath('toolbox'));
    if (~exist('paramFile','var'))
        paramFile = '../output/100.mat'; 
    end
    load(paramFile);
        
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

    disp('Read DevFile...');
    fid = fopen([params.data_path 'dev.txt'], 'r');
    file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
    fclose(fid);

    file_lines = file_lines{1};
    params.num_dev = size(file_lines, 1);

    dev_data = zeros(params.num_dev, 5);
    for i = 1 : params.num_dev
        s_tmp = regexp(file_lines(i), '\t', 'split');
        dev_data(i, 1) = entity_dict(s_tmp{1}{1});
        dev_data(i, 2) = relation_dict(s_tmp{1}{2});
        dev_data(i, 3) = entity_dict(s_tmp{1}{3});
        if (s_tmp{1}{4} == '1')
            dev_data(i, 4) = 1;
        else
            dev_data(i, 4) = -1;
        end
    end
    disp('Done.');

    disp('Read TestFile...');
    fid = fopen([params.data_path 'test.txt'], 'r');
    file_lines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
    fclose(fid);

    file_lines = file_lines{1};
    params.num_test = size(file_lines, 1);

    test_data = zeros(params.num_test, 5);
    for i = 1 : params.num_test
        s_tmp = regexp(file_lines(i), '\t', 'split');
        test_data(i, 1) = entity_dict(s_tmp{1}{1});
        test_data(i, 2) = relation_dict(s_tmp{1}{2});
        test_data(i, 3) = entity_dict(s_tmp{1}{3});
        if (s_tmp{1}{4} == '1')
            test_data(i, 4) = 1;
        else
            test_data(i, 4) = -1;
        end
    end
    disp('Done.');

    [W1, W2, b1, U, E] = stack2param(params.theta, params.decodeInfo);
    EE = zeros(params.embedding_size, params.num_entities);
    for e = 1 : params.num_entities
        EE(:,e) = mean(E(:, params.tree{e}.ids), 2);
    end
    E = EE;

    for i = 1 : params.num_dev
        e1 = dev_data(i, 1);
        e2 = dev_data(i, 3);
        rel = dev_data(i, 2);
        dev_data(i, 5) = 0;
        tmp_v = [E(:, e1); E(:, e2)];
        for k = 1 : params.slice_size
            dev_data(i, 5) = dev_data(i, 5) + U{rel}(k) * params.f(E(:, e1)' * W1{rel}(:,:,k) * E(:, e2) + W2{rel}(:, k)' * tmp_v + b1{rel}(k));
        end
    end

    lmax = min(dev_data(:, 5));
    rmax = max(dev_data(:, 5));
    best_threshold = zeros(params.num_relations, 1);
    best_acc = zeros(params.num_relations, 1);
    for i = 1 : params.num_relations
        best_threshold(i) = lmax;
        best_acc(i) = -1;
    end

    while (lmax <= rmax)
        for i = 1 : params.num_relations
            q = find(dev_data(:,2) == i);
            pred = (dev_data(q, 5) <= lmax) * 2 - 1;
            lmax_acc = mean(dev_data(q, 4) == pred);
            if (lmax_acc > best_acc(i))
                best_threshold(i) = lmax;
                best_acc(i) = lmax_acc;
            end
        end
        lmax = lmax + 0.01;
    end

    acc = 0;
    for i = 1 : params.num_test
        e1 = test_data(i, 1);
        e2 = test_data(i, 3);
        rel = test_data(i, 2);
    
        test_data(i, 5) = 0;
        tmp_v = [E(:, e1); E(:, e2)];
        for k = 1 : params.slice_size
            test_data(i, 5) = test_data(i, 5) + U{rel}(k) * params.f(E(:, e1)' * W1{rel}(:,:,k) * E(:, e2) + W2{rel}(:, k)' * tmp_v + b1{rel}(k));
        end

        if ((test_data(i, 5) <= best_threshold(rel) && test_data(i, 4) == 1) || ...
            (test_data(i, 5) > best_threshold(rel) && test_data(i, 4) == -1))
            acc = acc + 1;
        end
    end
    acc = acc / params.num_test;
    fprintf('Accuracy: %f\n', acc);