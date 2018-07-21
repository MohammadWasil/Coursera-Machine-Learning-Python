function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta is a column Vector, size(theta) = 2X1.

prediction = (X * theta) ;  		% size(X) = 97 X 2

meanSquare = (prediction - y).^2; 		% size(y) = 97 X 1

J =  sum(meanSquare) / (2*m);

% =========================================================================

end
