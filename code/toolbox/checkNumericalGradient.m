function maxDiff = checkNumericalGradient(J, theta)
	[cost, grad] = J(theta);
	numgrad = zeros(size(theta));
	mu = 1e-5;
	maxDiff = 0;
	for k = 1 : 20
			i = randi(length(theta));
			theta(i) = theta(i) + mu;
			val = J(theta);
			numgrad(i) = (val - cost) / mu;
			theta(i) = theta(i) - mu;
			fprintf('(%d/%d) %f %f %f\n', i, length(theta), grad(i), numgrad(i), abs(grad(i) - numgrad(i)));
			maxDiff = max(maxDiff, abs(grad(i) - numgrad(i)));
	end
end

