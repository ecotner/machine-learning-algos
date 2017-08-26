% initial data and relevant parameters
clear;
m = 1000;
n = 2;
d = 3;
learning_rate = .03;
regularization_parameter = 1000;
num_steps = 1000;
xy = (d/2)/sqrt(2);
X = [ones(m/2,1), randn(m/2,1)-xy, randn(m/2,1)-xy,;
     ones(m/2,1), randn(m/2,1)+xy, randn(m/2,1)+xy];
y = [zeros(m/2,1); ones(m/2,1)];
% add (3) quadratic features to X matrix (so X = [x0 x1 x2 x1^1 x2^2 x1*x2]
X = [X, X(:,2:3).^2, X(:,2).*X(:,3)];
n = n + 3;

% plotting data
figure(1);
clf;
scatter(X(1:m/2,2), X(1:m/2,3), 36, 'b');
hold on;
scatter(X(m/2+1:m,2), X(m/2+1:m,3), 36, 'r');
xlabel('x');
ylabel('y');
title('(x,y) positions of blue/red dots');
hold off;

% gradient descent algorithm
disp('Running gradient descent...');
init_theta = zeros(n+1,1);
[J_hist, theta] = GradientDescentRegularized(X, y, init_theta, learning_rate, regularization_parameter, num_steps);
disp('Done running, optimal theta is:');
theta

% plot time evolution of cost function
figure(2);
clf;
plot(1:length(J_hist),J_hist);
xlabel('Time step');
ylabel('J(\theta)');
title('Cost function vs. gradient descent timestep');

% plot decision boundary on scatter plot
x1 = (-(xy+5):0.1:(xy+5));
x2 = x1;
[XX, YY] = meshgrid(x1, x2);
% is there a way to do this that isn't by hand? seems like it requires a 3-dim array
%Z = theta(1) + theta(2)*XX + theta(3)*YY + theta(4)*XX.^2 + theta(5)*YY.^2 + theta(6)*XX.*YY;
Z = theta' * 
ZZ = 1./(1 + e.^(-Z));
figure(1);
hold on;
contour(XX, YY, ZZ, [0.5,0.5]);
hold off;
