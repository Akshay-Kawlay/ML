function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_temp = zeros(size(X,2),1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	for j =1:size(X,2)
		theta_temp(j) = theta(j) - (alpha/m)*sum((X*theta - y).*X(:,j),1);
		
	end
	for k = 1:size(X,2)
		theta(k) = theta_temp(k);
	end
	%fprintf('X(2)=%f\n', X(2));
	%fprintf('theta1=%f\n', theta1);
	%fprintf('theta2=%f\n', theta2);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	%fprintf('computeCost=%f\n', J_history(iter));
	
end

end
