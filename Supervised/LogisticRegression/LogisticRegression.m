% initial data and relevant parameters
clear;
m = 1000;
n = 2;
d = 2;
learning_rate = .03;
num_steps = 1000;
xy = (d/2)/sqrt(2);
X = [ones(m/2,1), randn(m/2,1)-xy, randn(m/2,1)-xy,;
     ones(m/2,1), randn(m/2,1)+xy, randn(m/2,1)+xy];
y = [zeros(m/2,1); ones(m/2,1)];     

% plotting data     
close;
figure(1);
scatter(X(1:m/2,2), X(1:m/2,3), 36, 'b');
hold on;
scatter(X(m/2:m,2), X(m/2:m,3), 36, 'r');
xlabel('x');
ylabel('y');
title('(x,y) positions of blue/red dots');
hold off;

% gradient descent algorithm
disp('Running gradient descent...');
init_theta = zeros(n+1,1);
[J_hist, theta] = GradientDescent(X, y, init_theta, learning_rate, num_steps);
disp('Done running, optimal theta is:');
theta

% plot time evolution of cost function
figure(2);
plot(1:length(J_hist),J_hist);
xlabel('Time step');
ylabel('J(\theta)');
title('Cost function vs. gradient descent timestep');

% plot decision boundary on scatter plot
x1 = (-5*xy:0.1:5*xy)';
x2 = (-1/theta(n+1))*[ones(length(x1),1) x1]*theta(1:n,:);
%x2 = (-1/theta(n+1)) * (theta(1)*ones(length(x1),1) + theta(2)*x1);
figure(1);
hold on;
plot(x1, x2);
hold off;