function cat_car_detection
clc
close all

        %% QUESTION 1 PART A
        disp('=== Question 1 Part A solution is initiated. ===')
        % import and read test and train datasets
        train_images = h5read('assign2_data1.h5','/trainims');
        train_images = double(train_images)/255;
        train_labels = h5read('assign2_data1.h5','/trainlbls');
        test_images = h5read('assign2_data1.h5','/testims');
        test_images = double(test_images)/255;
        test_labels = h5read('assign2_data1.h5','/testlbls');
        
        % find size of images, number of train images and test images
        [im_length, im_width, train_num] = size(train_images);
        [~,~,test_num] = size(test_images);
        % set hyperparameters
        eta = 0.2;
        batch_size = 32;
        epoch_num = 1000;
        N = 32;
        hyperparam = [eta, batch_size, epoch_num, N, 0];
        
        % train
        [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num);
        
        % plots
        figure
        plot(1:hyperparam(3), mse_train_epoch)
        hold on
        plot(1:hyperparam(3), mse_test_epoch)
        title( "MSE versus Epoch | N = " + hyperparam(4) + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "MSE")
        legend( 'Training' , 'Test' , 'Location' , 'northeast' )
        
        figure
        plot(1:hyperparam(3), class_error_train_epoch)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch)
        title( "Classification Error (%) versus Epoch | N = " + hyperparam(4) + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "Classification Error (%)")
        legend( 'Training' , 'Test' , 'Location' , 'northeast')
        
        %% QUESTION 1 PART C
        disp('=== Question 1 Part C solution is initiated. ===')
        % small number of hidden neurons
        hyperparam(4) = 8;
        % train
        [mse_test_epoch_low, mse_train_epoch_low, class_error_test_epoch_low, class_error_train_epoch_low] = cat_car_classifier(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num);
        % high number of hidden neurons
        hyperparam(4) = 256;
        % train
        [mse_test_epoch_high, mse_train_epoch_high, class_error_test_epoch_high, class_error_train_epoch_high] = cat_car_classifier(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num);
        
        % plots
        figure
        plot(1:hyperparam(3), mse_train_epoch)
        hold on
        plot(1:hyperparam(3), mse_test_epoch)
        hold on
        plot(1:hyperparam(3), mse_train_epoch_low)
        hold on
        plot(1:hyperparam(3), mse_test_epoch_low)
        hold on
        plot(1:hyperparam(3), mse_train_epoch_high)
        hold on
        plot(1:hyperparam(3), mse_test_epoch_high)
        title( "MSE versus Epoch | eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "MSE")
        legend( 'Training, N = 32' , 'Test, N = 32' , 'Training, N = 8' , 'Test, N = 8' , 'Training, N = 256' , 'Test, N = 256' , 'Location' , 'northeast' )
        
        figure
        plot(1:hyperparam(3), class_error_train_epoch)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch)
        hold on
        plot(1:hyperparam(3), class_error_train_epoch_low)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch_low)
        hold on
        plot(1:hyperparam(3), class_error_train_epoch_high)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch_high)
        title( "Classification Error (%) versus Epoch | eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "Classification Error (%)")
        legend( 'Training, N = 32' , 'Test, N = 32' , 'Training, N = 8' , 'Test, N = 8' , 'Training, N = 256' , 'Test, N = 256', 'Location' , 'northeast')
        
        %% QUESTION 1 PART D
        disp('=== Question 1 Part D solution is initiated. ===')
        hyperparam(4) = 32;
        hyperparam(5) = 32;
        
        % train
        [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier_2_layer(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num);
        
        % plots
        figure
        plot(1:hyperparam(3), mse_train_epoch)
        hold on
        plot(1:hyperparam(3), mse_test_epoch)
        title( "MSE versus Epoch | first layer = 32 , second layer = 32" + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "MSE")
        legend( 'Training' , 'Test' , 'Location' , 'northeast' )
        
        figure
        plot(1:hyperparam(3), class_error_train_epoch)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch)
        title( "Classification Error (%) versus Epoch | first layer = 32 , second layer = 32" + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2))
        xlabel( "Epoch Number" )
        ylabel( "Classification Error (%)")
        legend( 'Training' , 'Test' , 'Location' , 'northeast')
        
        %% QUESTION 1 PART E
        disp('=== Question 1 Part E solution is initiated. ===')
        alpha = 0.1;
        
        % train
        [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier_2_layer_momentum(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num, alpha); 
        
        % plots
        figure
        plot(1:hyperparam(3), mse_train_epoch)
        hold on
        plot(1:hyperparam(3), mse_test_epoch)
        title( "MSE versus Epoch | first layer = 32 , second layer = 32" + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2) + " , alpha = 0.1")
        xlabel( "Epoch Number" )
        ylabel( "MSE")
        legend( 'Training' , 'Test' , 'Location' , 'northeast' )
        
        figure
        plot(1:hyperparam(3), class_error_train_epoch)
        hold on
        plot(1:hyperparam(3), class_error_test_epoch)
        title( "Classification Error (%) versus Epoch | first layer = 32 , second layer = 32" + " , eta = " + hyperparam(1) + " , Batch Size = " + hyperparam(2) + " , alpha = 0.1")
        xlabel( "Epoch Number" )
        ylabel( "Classification Error (%)")
        legend( 'Training' , 'Test' , 'Location' , 'northeast')

end


%%
function [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num)

mse_train_epoch = zeros(1, hyperparam(3));
mse_test_epoch = zeros(1, hyperparam(3));
class_error_train_epoch = zeros(1, hyperparam(3));
class_error_test_epoch = zeros(1, hyperparam(3));

%Initilizaiton for one hidden layer NN
w_hidden = 0.001*randn(im_length*im_width, hyperparam(4));
b_hidden = 0.001*randn(hyperparam(4), 1);
w_output = 0.001*randn(hyperparam(4), 1);
b_output = zeros(1,1);

disp('Learning through epochs is initiated.')
for k = 1:hyperparam(3)
    [w_hidden, w_output, b_hidden, b_output, sum_error] = train_minibatch(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output);
    [~, accuracy_train] = calculate_error_accuracy(train_images, train_labels, w_hidden , w_output , b_hidden , b_output);
    class_error_train_epoch(k) = 100-accuracy_train;
    mse_train_epoch(k) = sum_error/length(train_images);
    
    [error_test, accuracy_test] = calculate_error_accuracy(test_images, test_labels, w_hidden , w_output , b_hidden , b_output);
    class_error_test_epoch(k) = 100-accuracy_test;
    mse_test_epoch(k) = error_test/length(test_images);
end

end

function [w_hidden, w_output, b_hidden, b_output, sum_error] = train_minibatch(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output)

sum_error = 0;

for k = 1:round(train_num/hyperparam(2))
    
    % Randomly place images to batches
    image_index = 1 + round((train_num-1) * rand(1, hyperparam(2)));
    batch_samples = zeros(im_length*im_width,hyperparam(2));
    batch_labels = zeros(1,hyperparam(2));
    for i = 1:hyperparam(2)
        batch_samples(:,i) = reshape(train_images(:,:,image_index(i)),[],1);
        batch_labels(i) = 2*train_labels(image_index(i)) -1;
    end
    
    % forward pass hidden layer
    v_hidden = w_hidden' * batch_samples + b_hidden;
    o_hidden = tanh(v_hidden);
    
    % forward pass output Layer
    v_output = w_output' * o_hidden + b_output;
    o_output = tanh(v_output);
    
    % backward pass output Layer
    local_gradient_output = (batch_labels-o_output).*(1-tanh(v_output).^2); % sigma_o
    del_w_output = -(o_hidden*(local_gradient_output)'); % output weights update
    del_b_output = -(local_gradient_output)*ones(hyperparam(2),1); % output bias update
    
    % backward pass hidden layer
    local_gradient_hidden = w_output*local_gradient_output.*(1-o_hidden.^2); % sigma_hidden
    del_w_hidden = -(batch_samples*local_gradient_hidden');
    del_b_hidden = -(local_gradient_hidden*ones(hyperparam(2), 1));
    
    % update output layer
    w_output = w_output-hyperparam(1)*del_w_output/hyperparam(2);
    b_output = b_output-hyperparam(1)*del_b_output/hyperparam(2);
    
    % update hidden layer
    w_hidden = w_hidden-hyperparam(1)*del_w_hidden/hyperparam(2);
    b_hidden = b_hidden-hyperparam(1)*del_b_hidden/hyperparam(2);
    
    sum_error = sum_error + sum((batch_labels - o_output).*(batch_labels - o_output));
end

end

function [error_test, accuracy_test] = calculate_error_accuracy(test_images, test_labels , w_hidden , w_output , b_hidden , b_output)
correct = 0;
error_test = 0;

% calculate accuracy and error for test set
for i = 1:length(test_images)
    input = reshape(test_images(:,:,i), [], 1);
    
    % forward pass
    v_hidden = w_hidden' * input + b_hidden;
    o_hidden = tanh(v_hidden);
    v_output = w_output' * o_hidden + b_output;
    o_output = tanh(v_output);
    
    if o_output*(2*test_labels(i)-1) > 0
        correct = correct + 1; % correctly classified
    end
    
    error_test = error_test + ((2*test_labels(i) - 1) - o_output).^2 ;
end
accuracy_test = 100 * correct / length(test_images);
end

function [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier_2_layer(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num)

mse_train_epoch = zeros(1, hyperparam(3));
mse_test_epoch = zeros(1, hyperparam(3));
class_error_train_epoch = zeros(1, hyperparam(3));
class_error_test_epoch = zeros(1, hyperparam(3));

%Initilizaiton for two hidden layer NN
w_hidden = 0.001*randn(im_length*im_width, hyperparam(4));
b_hidden = 0.001*randn(hyperparam(4), 1);
w_hidden_2 = 0.001*randn(hyperparam(4), hyperparam(5));
b_hidden_2 = 0.001*randn(hyperparam(5), 1);

w_output = 0.001*randn(hyperparam(5), 1);
b_output = zeros(1,1);

disp('Learning through epochs is initiated.')
for k = 1:hyperparam(3)
    [w_hidden, w_output, b_hidden, b_output, sum_error, w_hidden_2, b_hidden_2] = train_minibatch_2_layer(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output, w_hidden_2, b_hidden_2);
    [~, accuracy_train] = calculate_error_accuracy_2_layer(train_images, train_labels , w_hidden , w_output , b_hidden , b_output, w_hidden_2, b_hidden_2);
    class_error_train_epoch(k) = 100-accuracy_train;
    mse_train_epoch(k) = sum_error/length(train_images);
    
    [error_test, accuracy_test] = calculate_error_accuracy_2_layer(test_images, test_labels , w_hidden , w_output , b_hidden , b_output, w_hidden_2, b_hidden_2);
    class_error_test_epoch(k) = 100-accuracy_test;
    mse_test_epoch(k) = error_test/length(test_images);
end

end

function [w_hidden, w_output, b_hidden, b_output, sum_error, w_hidden_2, b_hidden_2] = train_minibatch_2_layer(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output, w_hidden_2, b_hidden_2)

sum_error = 0;

for k = 1:round(train_num/hyperparam(2))
    
    % Randomly place images to batches
    image_index = 1 + round((train_num-1) * rand(1, hyperparam(2)));
    batch_samples = zeros(im_length*im_width,hyperparam(2));
    batch_labels = zeros(1,hyperparam(2));
    for i = 1:hyperparam(2)
        batch_samples(:,i) = reshape(train_images(:,:,image_index(i)),[],1);
        batch_labels(i) = 2*train_labels(image_index(i)) -1;
    end
    
    % forward pass hidden layer
    v_hidden = w_hidden' * batch_samples + b_hidden;
    o_hidden = tanh(v_hidden);
    
    % forward pass hidden layer 2
    v_hidden_2 = w_hidden_2' * o_hidden + b_hidden_2;
    o_hidden_2 = tanh(v_hidden_2);
    
    % forward pass output Layer
    v_output = w_output' * o_hidden_2 + b_output;
    o_output = tanh(v_output);
    
    % backward pass output Layer
    local_gradient_output = (batch_labels-o_output).*(1-tanh(v_output).^2); % sigma_o
    del_w_output = -(o_hidden_2*(local_gradient_output)'); % output weights update
    del_b_output = -(local_gradient_output)*ones(hyperparam(2),1); % output bias update
    
    % backward pass hidden layer 2
    local_gradient_hidden_2 = w_output*local_gradient_output.*(1-o_hidden_2.^2); % sigma_hidden
    del_w_hidden_2 = -(o_hidden*local_gradient_hidden_2');
    del_b_hidden_2 = -(local_gradient_hidden_2*ones(hyperparam(2), 1));
    
    % backward pass hidden layer
    local_gradient_hidden = w_hidden_2*local_gradient_hidden_2.*(1-o_hidden.^2); % sigma_hidden
    del_w_hidden = -(batch_samples*local_gradient_hidden');
    del_b_hidden = -(local_gradient_hidden*ones(hyperparam(2), 1));
    
    % update output layer
    w_output = w_output-hyperparam(1)*del_w_output/hyperparam(2);
    b_output = b_output-hyperparam(1)*del_b_output/hyperparam(2);
    
    % update hidden layer 1
    w_hidden = w_hidden-hyperparam(1)*del_w_hidden/hyperparam(2);
    b_hidden = b_hidden-hyperparam(1)*del_b_hidden/hyperparam(2);
    
    % update hidden layer 2
    w_hidden_2 = w_hidden_2-hyperparam(1)*del_w_hidden_2/hyperparam(2);
    b_hidden_2 = b_hidden_2-hyperparam(1)*del_b_hidden_2/hyperparam(2);
    
   
    sum_error = sum_error + sum((batch_labels - o_output).*(batch_labels - o_output));
end

end

function [error_test, accuracy_test] = calculate_error_accuracy_2_layer(test_images, test_labels , w_hidden , w_output , b_hidden , b_output, w_hidden_2, b_hidden_2)
correct = 0;
error_test = 0;

% calculate accuracy and error for test set
for i = 1:length(test_images)
    input = reshape(test_images(:,:,i), [], 1);
    
    % forward pass
    v_hidden = w_hidden' * input + b_hidden;
    o_hidden = tanh(v_hidden);
    v_hidden_2 = w_hidden_2' * o_hidden + b_hidden_2;
    o_hidden_2 = tanh(v_hidden_2);
    v_output = w_output' * o_hidden_2 + b_output;
    o_output = tanh(v_output);
    
    if o_output*(2*test_labels(i)-1) > 0
        correct = correct + 1; % correctly classified
    end
    
    error_test = error_test + ((2*test_labels(i) - 1) - o_output).^2 ;
end
accuracy_test = 100 * correct / length(test_images);
end

function [mse_test_epoch, mse_train_epoch, class_error_test_epoch, class_error_train_epoch] = cat_car_classifier_2_layer_momentum(hyperparam, train_images, train_labels, test_images, test_labels, im_length, im_width, train_num, alpha)

mse_train_epoch = zeros(1, hyperparam(3));
mse_test_epoch = zeros(1, hyperparam(3));
class_error_train_epoch = zeros(1, hyperparam(3));
class_error_test_epoch = zeros(1, hyperparam(3));

%Initilizaiton for two hidden layer NN
w_hidden = 0.001*randn(im_length*im_width, hyperparam(4));
b_hidden = 0.001*randn(hyperparam(4), 1);
w_hidden_2 = 0.001*randn(hyperparam(4), hyperparam(5));
b_hidden_2 = 0.001*randn(hyperparam(5), 1);

w_output = 0.001*randn(hyperparam(5), 1);
b_output = zeros(1,1);

previous_w_hidden1 = 0;
previous_b_hidden1 = 0;
previous_w_hidden2 = 0;
previous_b_hidden2 = 0;
previous_w_out = 0;
previous_b_out = 0;
        
disp('Learning through epochs is initiated.')
for k = 1:hyperparam(3)
    [w_hidden, w_output, b_hidden, b_output, sum_error, w_hidden_2, b_hidden_2, previous_w_out, previous_b_out, previous_w_hidden1, previous_b_hidden1, previous_w_hidden2, previous_b_hidden2] = train_minibatch_2_layer_momentum(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output, w_hidden_2, b_hidden_2, previous_w_out, previous_b_out, previous_w_hidden1, previous_b_hidden1, previous_w_hidden2, previous_b_hidden2, alpha);
    [~, accuracy_train] = calculate_error_accuracy_2_layer(train_images, train_labels , w_hidden , w_output , b_hidden , b_output, w_hidden_2, b_hidden_2);
    class_error_train_epoch(k) = 100-accuracy_train;
    mse_train_epoch(k) = sum_error/length(train_images);
    
    [error_test, accuracy_test] = calculate_error_accuracy_2_layer(test_images, test_labels , w_hidden , w_output , b_hidden , b_output, w_hidden_2, b_hidden_2);
    class_error_test_epoch(k) = 100-accuracy_test;
    mse_test_epoch(k) = error_test/length(test_images);
end

end

function [w_hidden, w_output, b_hidden, b_output, sum_error, w_hidden_2, b_hidden_2, previous_w_out, previous_b_out, previous_w_hidden1, previous_b_hidden1, previous_w_hidden2, previous_b_hidden2] = train_minibatch_2_layer_momentum(hyperparam, train_images, train_labels, im_length, im_width, train_num, w_hidden, w_output, b_hidden, b_output, w_hidden_2, b_hidden_2, previous_w_out, previous_b_out, previous_w_hidden1, previous_b_hidden1, previous_w_hidden2, previous_b_hidden2, alpha)

sum_error = 0;

for k = 1:round(train_num/hyperparam(2))
    
    % Randomly place images to batches
    image_index = 1 + round((train_num-1) * rand(1, hyperparam(2)));
    batch_samples = zeros(im_length*im_width,hyperparam(2));
    batch_labels = zeros(1,hyperparam(2));
    for i = 1:hyperparam(2)
        batch_samples(:,i) = reshape(train_images(:,:,image_index(i)),[],1);
        batch_labels(i) = 2*train_labels(image_index(i)) -1;
    end
    
    % forward pass hidden layer
    v_hidden = w_hidden' * batch_samples + b_hidden;
    o_hidden = tanh(v_hidden);
    
    % forward pass hidden layer 2
    v_hidden_2 = w_hidden_2' * o_hidden + b_hidden_2;
    o_hidden_2 = tanh(v_hidden_2);
    
    % forward pass output Layer
    v_output = w_output' * o_hidden_2 + b_output;
    o_output = tanh(v_output);
    
    % backward pass output Layer
    local_gradient_output = (batch_labels-o_output).*(1-tanh(v_output).^2); % sigma_o
    del_w_output = -(o_hidden_2*(local_gradient_output)'); % output weights update
    del_b_output = -(local_gradient_output)*ones(hyperparam(2),1); % output bias update
    
    % backward pass hidden layer 2
    local_gradient_hidden_2 = w_output*local_gradient_output.*(1-o_hidden_2.^2); % sigma_hidden
    del_w_hidden_2 = -(o_hidden*local_gradient_hidden_2');
    del_b_hidden_2 = -(local_gradient_hidden_2*ones(hyperparam(2), 1));
    
    % backward pass hidden layer
    local_gradient_hidden = w_hidden_2*local_gradient_hidden_2.*(1-o_hidden.^2); % sigma_hidden
    del_w_hidden = -(batch_samples*local_gradient_hidden');
    del_b_hidden = -(local_gradient_hidden*ones(hyperparam(2), 1));
    
    % update output layer WITH MOMENTUM
    w_output = w_output-hyperparam(1)*del_w_output/hyperparam(2) + alpha * previous_w_out;
    b_output = b_output-hyperparam(1)*del_b_output/hyperparam(2) + alpha * previous_b_out;
    
    % update hidden layer 1 WITH MOMENTUM
    w_hidden = w_hidden-hyperparam(1)*del_w_hidden/hyperparam(2) + alpha * previous_w_hidden1;
    b_hidden = b_hidden-hyperparam(1)*del_b_hidden/hyperparam(2) + alpha * previous_b_hidden1;
    
    % update hidden layer 2 WITH MOMENTUM
    w_hidden_2 = w_hidden_2-hyperparam(1)*del_w_hidden_2/hyperparam(2) + alpha * previous_w_hidden2;
    b_hidden_2 = b_hidden_2-hyperparam(1)*del_b_hidden_2/hyperparam(2) + alpha * previous_b_hidden2;
    
    % store past updates WITH MOMENTUM
    previous_w_out = -hyperparam(1)*del_w_output/hyperparam(2) + alpha * previous_w_out;
    previous_b_out = -hyperparam(1)*del_b_output/hyperparam(2) + alpha * previous_b_out;
    
    previous_w_hidden1 = -hyperparam(1)*del_w_hidden/hyperparam(2) + alpha * previous_w_hidden1;
    previous_b_hidden1 = -hyperparam(1)*del_b_hidden/hyperparam(2) + alpha * previous_b_hidden1;
    
    previous_w_hidden2 = -hyperparam(1)*del_w_hidden_2/hyperparam(2) + alpha * previous_w_hidden2;
    previous_b_hidden2 = -hyperparam(1)*del_b_hidden_2/hyperparam(2) + alpha * previous_b_hidden2;

    sum_error = sum_error + sum((batch_labels - o_output).*(batch_labels - o_output));
end

end

