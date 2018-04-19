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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%vectorize y
temp=zeros(num_labels,m);
for i=1:m
    temp(y(i),i)=1;
end
y=temp;
%%%%%%
X_adda0=[ones(m,1) X];
a2=sigmoid(Theta1*X_adda0')';
a2_adda0=[ones(m,1) a2];
a3=sigmoid(Theta2*a2_adda0');
h=a3;
temp=0;
for i=1:m
    for k=1:num_labels
        temp=temp+(y(k,i)*log(h(k,i))+(1-y(k,i))*log(1-(h(k,i))));
    end
end
temp=-temp/m;
theta_square=sum(sum(Theta1(1:hidden_layer_size,2:input_layer_size+1).^2))+sum(sum(Theta2(1:num_labels,2:hidden_layer_size+1).^2));
J=temp+lambda/(2*m)*theta_square;

delta3=a3-y;
delta2=(Theta2)'*delta3.*(a2_adda0.*(1-a2_adda0))';


DELTA1=0;
DELTA2=0;
DELTA1=DELTA1+delta2*X_adda0;
DELTA2=DELTA2+delta3*a2_adda0;


Theta1_grad=1/m*DELTA1(2:hidden_layer_size+1,:)+lambda/m*Theta1;
Theta1_grad(:,1)=Theta1_grad(:,1)-lambda/m*Theta1(:,1);
Theta2_grad=1/m*DELTA2+lambda/m*Theta2;
Theta2_grad(:,1)=Theta2_grad(:,1)-lambda/m*Theta2(:,1);












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
