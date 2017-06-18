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

 X = [ones(m, 1) X];
 %
 

 %Re-iterating a previous tip... 
 %if you have a matrix A = [ 1 2 3 4 5 6 7 8 9 10 ],
 %and a scalar c = 4, 
 %the expression A == c will yield a vector of dimension size(A) [ 10, in this case ].
 %In this example A == c is [ 0 0 0 1 0 0 0 0 0 0 ]

%  Would appreciate confirmation or correction of my understanding, specifically about data structure and indexing.
%In regards to the cost function:
% y(i)k: guessing it is the i th row of the y column vector converted to a 10 vector representation of the digit.
%for example if i th row is 5, then [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ]?
% a(3): 5000 by 10 matrix?
% a(3)k: is the kth column of the the 5000 by 10 matrix? SO ==> yv(:,k)' log(hX3(:,k))

 hX2=  sigmoid(X*Theta1');
 
 %%%  hX(2)        5000 * 25
 z2=hX2;

 hX2 = [ones(m,1) hX2];
 a2=hX2;

 hX3=sigmoid(hX2*Theta2');
 z3=hX3;
 a3=z3;
 yv = repmat(1:num_labels, size(y,1) , 1) == repmat(y, 1, num_labels);
 costSum=0;

 for k=1:num_labels
     
     costSum =costSum+( (yv(:,k)'*log(hX3(:,k))) + (1 - yv(:,k)')*log (1 - hX3(:,k)));
 end
     sumTheta1=0;
    
    sumTheta2=0;

    for jT1=1:size(Theta1,1)
        
        for kT1=2:(size(Theta1,2))
                
           sumTheta1 =sumTheta1+(Theta1(jT1,kT1).^2);
            
        end
    
    end 
    
    
    for jT2=1:size(Theta2,1)
        
        for kT2=2:(size(Theta2,2))
                
           sumTheta2 =sumTheta2+(Theta2(jT2,kT2).^2);
            
        end
    
    end 
    %lambda=1;
   regularParam=(lambda/(2*m))*(sumTheta1+sumTheta2);
   
   J = ((-1/m) * costSum)+regularParam;        
    
    
    a1=X;
    delta3= a3- yv;  % 5000 * 10
    
    %Theta2(:,1)= [];
    
    %theta2     10 * 26
    r2 = delta3*Theta2;   % 5000 * 25
    delta2=r2.*(a2.*(1-a2));
    % delta2 = sigmoidGradient(z2).*r2;   % 5000 * 25    
    %remove first column  
    % all rows in first column  === [] null
    delta2(:,1)= [];
    
    if J==0
        Theta1_grad = Theta1_grad+(1/m).*( delta2' * a1);   
        Theta2_grad = Theta2_grad+(1/m).*(delta3' * a2);
        % Unroll gradients
        grad = [Theta1_grad(:) ; Theta2_grad(:)];
    else
         t1 = (lambda/m).*Theta1;
         t1(:,1)= zeros;
         Theta1_grad = Theta1_grad+(1/m).*( delta2' * a1) + t1;   
         t2 = (lambda/m).*Theta2;
         t2(:,1)= zeros;
         Theta2_grad = Theta2_grad+(1/m).*(delta3' * a2) + t2;   
         % Unroll gradients
         grad = [Theta1_grad(:) ; Theta2_grad(:)];
    end
    
    


    %%%%%%%%%%%%%%%%%%%% FOR PART 1 JUST THIS LINE WITHOUT OTHER LINES J = ((-1/m) * costSum);
   
    
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

%%%%%%%%%%%%%%%%%%                               Partial Derivetive                           %%%%%%%%%%%%%%%%%

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------------------------------------------------

% =========================================================================



end
