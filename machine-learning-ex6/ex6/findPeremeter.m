load('ex6data3.mat');
C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
sigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
model= svmTrain(X, y, C(5), @(x1, x2) gaussianKernel(x1, x2, sigma(3)));
visualizeBoundary(X, y, model);
C(5)
sigma(3)







