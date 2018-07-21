function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Cowith regularizationmpute cost and gradient for logistic regression 
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

shift_theta = theta(2:size(theta));
theta_reg = [0;shift_theta];

J = (1/m)*(-y'* log(h) - (1 - y)'*log(1-h))+(lambda/(2*m))*theta_reg'*theta_reg;



%J = ((-((y)' * log(h)) - ((1-y)'* log(1-h))) * (1/m) ) + ((theta.^2) * (lambda / (2*m)));

grad = ((1/m) * X' * (h - y)) + ((lambda/m) * theta_reg);


% =============================================================

end
