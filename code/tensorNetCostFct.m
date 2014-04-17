function [cost, grad] = tensorNetCostFct(theta, data_batch, params, flipType)
    [W1, W2, b1, U, E] = stack2param(theta, params.decodeInfo);
    cost = 0;
    update = 0;

    entVec = zeros(params.embedding_size, params.num_entities);
    entVecGrad = zeros(params.embedding_size, params.num_entities);

    Eent = cell(params.num_entities, 1);
    for e = 1 : params.num_entities
        entVec(:,e) = mean(E(:, params.tree{e}.ids), 2);
    end

    for i = 1 : params.num_relations
        lst = data_batch.rel == i;
        
        mAll = sum(lst);
        e1 = data_batch.e1(lst);
        e2 = data_batch.e2(lst);
        e3 = data_batch.e3(lst);
        entVecE1 = entVec(:, e1);
        entVecE2=  entVec(:, e2);
        entVecE3 = entVec(:, e3);
        if flipType
            % replacing right entities with random ones
            entVecE1Neg = entVecE1;
            entVecE2Neg = entVecE3;
            e1Neg = e1;
            e2Neg = e3;
        else
            % replacing left entities with random ones
            entVecE1Neg = entVecE3;
            entVecE2Neg = entVecE2;
            e1Neg = e3;
            e2Neg = e2;
        end
        
        
        v_pos = zeros(params.slice_size, mAll);
        for k = 1 : params.slice_size
            v_pos(k, :) = sum(bsxfun(@times, entVecE1, W1{i}(:, :, k) * entVecE2));
        end
        if params.inTensorKeepNormal
            v_pos = v_pos + bsxfun(@plus, b1{i}', W2{i}' * [entVecE1; entVecE2]);
        end
        
        v_neg = zeros(params.slice_size, mAll);
        for k = 1 : params.slice_size
            v_neg(k, :) = sum(bsxfun(@times, entVecE1Neg, W1{i}(:, :, k) * entVecE2Neg));
        end
        if params.inTensorKeepNormal
            v_neg = v_neg + bsxfun(@plus, b1{i}', W2{i}' * [entVecE1Neg; entVecE2Neg]);
        end
        
        z_pos = params.f(v_pos);
        z_neg = params.f(v_neg);
        
        score_pos = U{i}'*z_pos;
        score_neg = U{i}'*z_neg;
        
        indx = (score_pos + 1 > score_neg);
        cost = cost + sum(indx .* (score_pos + 1 - score_neg));
        
        update = update + sum(indx);
        
        gradb1{i} = zeros(size(b1{i}));
        gradW1{i} = zeros(size(W1{i}));
        gradW2{i} = zeros(size(W2{i}));
        
        
        % filter for only active
        m = sum(indx);
        z_pos=z_pos(:,indx);
        z_neg=z_neg(:,indx);
        entVecE1Rel = entVecE1(:,indx);
        entVecE2Rel = entVecE2(:,indx);
        entVecE1RelNeg = entVecE1Neg(:,indx);
        entVecE2RelNeg = entVecE2Neg(:,indx);
        e1 = e1(indx);
        e2 = e2(indx);
        e1Neg = e1Neg(indx);
        e2Neg = e2Neg(indx);
        
        gradU{i} = sum(z_pos - z_neg,2);
        
        tmp_posAll = bsxfun(@times,U{i}, params.df(z_pos));
        tmp_negAll = - bsxfun(@times, U{i}, params.df(z_neg));
        
        gradb1{i} = sum(tmp_posAll + tmp_negAll,2);
        
        for k = 1 : params.slice_size
            tmp_pos=tmp_posAll(k,:);
            tmp_neg=tmp_negAll(k,:);
            
            gradW1{i}(:, :, k) = bsxfun(@times, entVecE1Rel, tmp_pos) * entVecE2Rel' + bsxfun(@times, entVecE1RelNeg, tmp_neg) * entVecE2RelNeg';
               
            if params.inTensorKeepNormal
                gradW2{i}(:, k) = sum(bsxfun(@times, [entVecE1Rel; entVecE2Rel], tmp_pos) + ...
                    bsxfun(@times, [entVecE1RelNeg; entVecE2RelNeg], tmp_neg), 2);
                V_pos = bsxfun(@times, W2{i}(:, k), tmp_pos);
                V_neg = bsxfun(@times, W2{i}(:, k), tmp_neg);
                
                entVecGrad = entVecGrad + ....
                    V_pos(1:params.embedding_size,:) * sparse(1:m, e1, ones(m,1), m, params.num_entities) + ...
                    V_pos(params.embedding_size + 1:end, :) * sparse(1:m, e2, ones(m,1), m, params.num_entities) + ...
                    V_neg(1:params.embedding_size,:) * sparse(1:m, e1Neg, ones(m,1), m, params.num_entities) + ...
                    V_neg(params.embedding_size + 1:end, :) * sparse(1:m, e2Neg, ones(m,1), m, params.num_entities);
            end
            
            entVecGrad = entVecGrad + ...
                bsxfun(@times, W1{i}(:,:,k) * entVec(:,e2), tmp_pos) * sparse(1:m, e1, ones(m,1), m, params.num_entities) + ...
                bsxfun(@times, W1{i}(:,:,k)' * entVec(:,e1), tmp_pos) * sparse(1:m, e2, ones(m,1), m, params.num_entities) + ...
                bsxfun(@times, W1{i}(:,:,k) * entVec(:,e2Neg), tmp_neg) * sparse(1:m, e1Neg, ones(m,1), m, params.num_entities) + ...
                bsxfun(@times, W1{i}(:,:,k)' * entVec(:,e1Neg), tmp_neg) * sparse(1:m, e2Neg, ones(m,1), m, params.num_entities);
        end
        
        gradW1{i} = gradW1{i} ./ params.batch_size;
        gradW2{i} = gradW2{i} ./ params.batch_size;
        gradb1{i} = gradb1{i} ./ params.batch_size;
        gradU{i} = gradU{i} ./ params.batch_size;
    end

    gradE = zeros(size(E));
    for e = 1 : params.num_entities
        sl = params.tree{e}.num;
        gradE(:, params.tree{e}.ids) = gradE(:, params.tree{e}.ids) + repmat(entVecGrad(:, e) / sl, 1, sl);
    end

    gradE = gradE ./ params.batch_size;
    cost = cost ./ params.batch_size;

    if ~params.inTensorKeepNormal
        for i=1:length(gradW2)
            gradW2{i} = zeros(size(gradW2{i}));
            gradb1{i} = zeros(size(gradb1{i}));
            gradU{i} = zeros(size(U{i}));
        end 
    end
    display(['incorrect = ' num2str(update) '/' num2str(params.batch_size * params.corrupt_size)]);
    grad = param2stack(gradW1, gradW2, gradb1, gradU, gradE);

    cost = cost + params.reg_parameter / 2 * sum(theta.^2);
    grad = grad + params.reg_parameter * theta;
end

