function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%initialize activation units
%a2 = zeros(size(Theta1,1), 1);
%a3 = zeros(size(Theta2,1), 1);

%add row of 1 to X
X = [ones(m,1) X];

%get y in proper format
y_new = zeros(m,1,num_labels);
logical_array = [1:num_labels]';
for i = 1:m
	y_new(i,:) = logical_array == y(i);
end
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%forward propagation
for i = 1:m
	z2 = X(i,:)*Theta1';
	a2 = sigmoid(z2);
	a2 = [1 a2];
	z3 = a2*Theta2';
	a3 = sigmoid(z3); %this is h(x)
	J = J + sum((-y_new(i,:)).*log(a3) - (ones(1,num_labels)-y_new(i,:)).*log(1-a3), 2);
end
J = (1/m)*J;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Big_delta1 = 0;
Big_delta2 = 0;
for i = 1:m
	z2 = X(i,:)*Theta1';
	a2 = sigmoid(z2);
	a2 = [1 a2];
	z3 = a2*Theta2';
	a3 = sigmoid(z3); %this is h(x)
	delta3 = a3 - y_new(i,:);
	%u = sigmoidGradient([1 z2]);
	%size(((Theta2')*delta3'))
	%size(u');
	delta2 = ((Theta2')*delta3').*sigmoidGradient([1 z2]');
	%size(delta2)
	delta2 = delta2(2:end);
	Big_delta1 = Big_delta1 + delta2*X(i,:);
	Big_delta2 = Big_delta2 + (delta3')*a2;
end

grad1_reg = Theta1;
grad2_reg = Theta2;
grad1_reg(:,1) = 0;
grad2_reg(:,1) = 0;
Theta1_grad = Big_delta1/m + lambda*grad1_reg/m;
Theta2_grad = Big_delta2/m + lambda*grad2_reg/m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
% Calculate regularization term
sumJ = 0;
for i = 1:size(Theta1,1)
	val1 = Theta1(i,2:end)*(Theta1(i,2:end)');
	sumJ = sumJ + val1;
end
for i = 1:size(Theta2,1)
	val2 = Theta2(i,2:end)*(Theta2(i,2:end)');
	sumJ = sumJ + val2;
end

reg_J = (lambda/(2*m))*sumJ;
J = J + reg_J;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
