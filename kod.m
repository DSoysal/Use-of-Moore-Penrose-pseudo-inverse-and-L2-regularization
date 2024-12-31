%% LOAD THE DATASET
load("mnist.mat")

idx = 23053;

x = reshape (X(: , idx) , [28 28]).' ;
label = labels (idx) ;
%% VISUALIZATION - part a)
figure;
imshow(x,[]);
title(["Digit label: ", num2str(label)]);
%% BLURRING - part b) c)
n = -2:2;

Z = 0; % normalization constant
for m1 = n
    for m2 = n
        Z = Z + exp(-(m1^2+m2^2)/2);
    end
end

K = zeros(5, 5);

for n1 = n % kernel matrix
    for n2 = n
        K(n1+3, n2+3) = 1/Z*exp(-(n1^2+n2^2)/2);
    end
end

y = conv2(x, K);

figure;
imshow(y,[]);
title(["Blurred image: ", num2str(label)]);
%% OBTAINING A - part d)
x_vec = x(:);

[height_x, width_x] = size(x);
x_vec_size = numel(x_vec);
y_vec_size = numel(y(:));

A = zeros(y_vec_size, x_vec_size);

for i = 1:x_vec_size
    temp_x = zeros(height_x, width_x);
    temp_x(i) = 1; 

    temp_y = conv2(temp_x, K); 

    A(:, i) = temp_y(:)';
end

y_vec = A * x_vec;
norm(y - reshape(y_vec , [32 32]))
imshow(reshape(y_vec , [32 32]),[]);
%% OBTAINING X WITH PSEUDO INVERSE part e)
x_mnls_tilde = pinv(A)*y_vec;
imshow(reshape(x_mnls_tilde,  [height_x width_x]), []);

diff_image = x - reshape(x_mnls_tilde, [height_x, width_x]);
mse = mean(diff_image(:).^2);
%% ADDING NOISE - part f)
w = randn(size(y)); 
y_tilde = y + w;

figure;
imshow(y, []);
title('Original Blurred Image (y)');

figure;
imshow(y_tilde, []);
title('Noisy Blurred Image (ỹ)');

numerator = norm(y_tilde(:) - y(:));
denominator = norm(y(:));
normalized_difference = numerator / denominator
%% OBTAINING X WITH NOISE USING PSEUDO INVERSE - part g)
x_mnls_tilde = pinv(A)*y_tilde(:);
imshow(reshape(x_mnls_tilde,  [height_x width_x]), []);

diff_image = x - reshape(x_mnls_tilde, [height_x, width_x]);
mse = mean(diff_image(:).^2)
%% SINGULAR VALUE OBSERVATION - part h)
[U, S, V] = svd(A);
singular_values = diag(S);
singular_values_A_pseudo = 1 ./ singular_values;
k_A = max(singular_values) / min(singular_values);

fprintf('Condition Number (kappa(A)): %e\n', k_A);
fprintf('Singular Values of A: \n');
disp(singular_values);
fprintf('Singular Values of A† (Pseudo-Inverse): \n');
disp(singular_values_A_pseudo);
%% REGULARIZATION part - i)
lambda = 0.01; % best = 0.0013

x_reg = (A' * A + lambda * eye(size(A, 2))) \ (A' * y_tilde(:));

x_reg_image = reshape(x_reg, [height_x, width_x]);
figure;
imshow(x_reg_image, []);
title(['Regularized Solution (λ = ', num2str(lambda), ')']);

reg_error = norm(x_reg - x(:)) / norm(x(:))
%% DOWNSAMPLING part j)
z = y(1 : 2 : end, 1 : 2 : end);

figure;
imshow(z,[]);
%% SAME WITH Z part k)
[height_z, width_z] = size(z);
z_vec_size = numel(z);

B = zeros(z_vec_size, x_vec_size);

for i = 1:x_vec_size
    temp_x = zeros(height_x, width_x);
    temp_x(i) = 1; 

    temp_z = conv2(temp_x, K);
    temp_z = temp_z(1 : 2 : end, 1 : 2 : end);

    B(:, i) = temp_z(:)';
end

z_vec = B * x_vec;
norm(z - reshape(z_vec , [16 16]))
imshow(reshape(z_vec , [16 16]),[]);
%% OBTAINING X WITH PSEUDO INVERSE - part k) continued
x_mnls_tilde = pinv(B)*z_vec;
imshow(reshape(x_mnls_tilde,  [height_x width_x]), []);

diff_image = x - reshape(x_mnls_tilde, [height_x, width_x]);
mse = mean(diff_image(:).^2);