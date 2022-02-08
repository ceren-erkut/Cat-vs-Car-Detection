function ceren_erkut_21602906_hw2(question)
clc
close all
switch question
    case '1'
	disp('1')
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
        
    case '2'
	disp('2')
        %% QUESTION 2 PART A & B
        disp('=== Question 2 solution is initiated. ===')
               
        testd = h5read('assign2_data2.h5','/testd');
        testx = h5read('assign2_data2.h5','/testx');
        traind = h5read('assign2_data2.h5','/traind');
        trainx = h5read('assign2_data2.h5','/trainx');
        vald = h5read('assign2_data2.h5','/vald');
        valx = h5read('assign2_data2.h5','/valx');
        words = h5read('assign2_data2.h5','/words');
        index_sample = randperm(length(trainx));
        trainx = trainx( :,index_sample);
        traind = traind(index_sample, :);
        words = string(words);
        
        % hyperparameters
        hyperparam = zeros(1,7);
        hyperparam(1) = 200; % batch size
        hyperparam(2) = 250; % vocabulary length
        hyperparam(3) = 0.85; % momentum
        hyperparam(4) = 0.15; % learning rate
        hyperparam(5) = 50; % epoch max
        hyperparam(6) = 256; % P
        hyperparam(7) = 32; % D
        sample_index_table = randi( [1, length(testx)] , 1,5 );
        
        tic
        [w_output, b_output, w_hidden, b_hidden, embed]= nlp(hyperparam, trainx, traind, vald, valx);
        [accuracy_1] = triagram_list(w_output, b_output, w_hidden, b_hidden, embed, hyperparam, testx, testd, sample_index_table, words);
        disp( "Pair (D = 32, P = 256): Accuracy = " + accuracy_1);
        elapsed_time = toc / 60;
        display("Pair (D = 32, P = 256): Elapsed time = " + elapsed_time + " minutes")
        hyperparam(6) = 128; % P
        hyperparam(7) = 16; % D
        tic
        [w_output, b_output, w_hidden, b_hidden, embed]= nlp(hyperparam, trainx, traind, vald, valx);
        [accuracy_2] = triagram_list(w_output, b_output, w_hidden, b_hidden, embed, hyperparam, testx, testd, sample_index_table, words);
        disp( "Pair (D = 16, P = 128): Accuracy = " + accuracy_2);
        elapsed_time = toc / 60;
        display("Pair (D = 16, P = 128): Elapsed time = " + elapsed_time + " minutes")
        hyperparam(6) = 64; % P
        hyperparam(7) = 8; % D
        tic
        [w_output, b_output, w_hidden, b_hidden, embed]= nlp(hyperparam, trainx, traind, vald, valx);
        [accuracy_3] = triagram_list(w_output, b_output, w_hidden, b_hidden, embed, hyperparam, testx, testd, sample_index_table, words);
        disp( "Pair (D = 8, P = 64): Accuracy = " + accuracy_3);
        elapsed_time = toc / 60;
        display("Pair (D = 8, P = 64): Elapsed time = " + elapsed_time + " minutes")
    case '3'
	disp('3')
        %% QUESTION 3
        disp('=== Question 3 solution is in the report. ===')
end

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

function [w_output, b_output, w_hidden, b_hidden, embed]= nlp(hyperparam, trainx, traind, vald, valx)
stop_condition = 1;
previous_w_output = 0;
previous_b_output = 0;
previous_w_hidden = 0;
previous_b_hidden = 0;
previous_embed = 0;

embed = normrnd(0, 1, [hyperparam(2), hyperparam(7)]);
w_output = normrnd(0, 0.01, [hyperparam(6), hyperparam(2)]);
b_output = normrnd(0, 0.01, [hyperparam(2), 1]);
w_hidden = normrnd(0, 0.01, [3*hyperparam(7), hyperparam(6)]);
b_hidden = normrnd(0, 0.01, [hyperparam(6), 1]);

training_cross_entropy = zeros(1, hyperparam(5));
test_cross_entropy = zeros(1, hyperparam(5));

disp( "Pair (D = " + hyperparam(7) + ", P = " + hyperparam(6) + "): Training starts.");
% epoch training
for i=1:hyperparam(5)
    error = 0;
    [w_output, b_output, w_hidden, b_hidden, previous_w_output, previous_b_output, previous_w_hidden, previous_b_hidden, previous_embed, hyperparam, embed, epoch_loss] = nlp_train_epoch(trainx, traind, w_output, b_output, w_hidden, b_hidden, previous_w_output, previous_b_output, previous_w_hidden, previous_b_hidden, previous_embed, embed, hyperparam); 
    % one hot encoding
    for k = 1:length(valx)
        sample_word = valx(:,k);
        target_word = zeros(hyperparam(2), 1);
        target_word(vald(k)) = 1;        
        % 3 bit encoding
        one_hot_first = zeros(hyperparam(2), 1);
        one_hot_first(sample_word(1),1) = 1;
        one_hot_second = zeros(hyperparam(2), 1);
        one_hot_second(sample_word(2),1) = 1;
        one_hot_third = zeros(hyperparam(2), 1);
        one_hot_third(sample_word(3),1) = 1;
        pre_word = embed'*[one_hot_first,one_hot_second,one_hot_third ];
        sample_word = reshape(pre_word,3*hyperparam(7),1);       
        % forward pass
        v_hidden = b_hidden+w_hidden'*sample_word;
        o_hidden = ones(hyperparam(6),1)./(1+exp(-v_hidden));
        v_output = b_output+w_output'*o_hidden;
        o_output = exp(v_output)./sum(exp(v_output));        
        error = error + abs(sum(target_word.*log(o_output)));
    end    
    error = error/length(valx);
    % stop condition
    if(i ~= 1 && (error_previous - error) < 0.0002)
        stop_condition = 0;
    end    
    error_previous = error;
    test_cross_entropy(i) = error;
    training_cross_entropy(i) = epoch_loss;
    % plot errors
    figure(hyperparam(6)/64)
    plot(1:i, training_cross_entropy(1:i), 'b');
    hold on
    plot(1:i, test_cross_entropy(1:i), 'r');
    legend('Training Set', 'Validation Set');
    title("Cross Entropy Error vs. Epoch | P = " +  hyperparam(6) + ", D = " + hyperparam(7));
    xlabel('Epoch');
    ylabel('Cross Entropy Error');
    grid on
    if stop_condition == 0
        break;
    end
end
end

function [w_output, b_output, w_hidden, b_hidden, previous_w_output, previous_b_output, previous_w_hidden, previous_b_hidden, previous_embed, hyperparam, embed, epoch_error] = nlp_train_epoch(trainx, traind, w_output, b_output, w_hidden, b_hidden, previous_w_output, previous_b_output, previous_w_hidden, previous_b_hidden, previous_embed, embed, hyperparam)
bound = ceil(length(trainx)/hyperparam(1));
epoch_error = zeros(ceil(length(trainx)/hyperparam(1)),1);

for k=1:bound
    
    training_error = 0;
    if k < bound
        batch = trainx(:,(k-1)*hyperparam(1)+1:(k)*hyperparam(1));
        batch_output = traind((k-1)*hyperparam(1)+1:(k)*hyperparam(1));
    else
        batch = trainx(:,(k-1)*hyperparam(1)+1:end);
        batch_output = traind((k-1)*hyperparam(1)+1:end);
    end
    
    [training_error, del_w_out, del_b_out, del_w_hidden, del_b_hidden, del_embed] = nlp_train_minibatch(batch ,batch_output, hyperparam, embed, w_output , b_output , w_hidden , b_hidden , training_error);  
    
    training_error = training_error/hyperparam(1);
    epoch_error(k) = training_error;
    previous_w_output = -hyperparam(4)*del_w_out/hyperparam(1) + hyperparam(3) * previous_w_output;
    previous_b_output = -hyperparam(4)*del_b_out/hyperparam(1) + hyperparam(3) * previous_b_output;
    previous_w_hidden = -hyperparam(4)*del_w_hidden/hyperparam(1) + hyperparam(3) * previous_w_hidden;
    previous_b_hidden = -hyperparam(4)*del_b_hidden/hyperparam(1) + hyperparam(3) * previous_b_hidden;
    
    previous_embed = -hyperparam(4)*del_embed/hyperparam(1) + hyperparam(3)*previous_embed;
     
    w_output = w_output + previous_w_output;
    b_output = b_output + previous_b_output;
    w_hidden = w_hidden +previous_w_hidden;
    b_hidden = b_hidden +previous_b_hidden;
    embed = embed + previous_embed;
    
end
epoch_error = sum(epoch_error) / length(epoch_error);

end

function [training_error, del_w_out, del_b_out, del_w_hidden, del_b_hidden, del_embed]= nlp_train_minibatch(batch ,batch_output, hyperparam, embed, w_output , b_output , w_hidden , b_hidden , training_error)
del_w_out = zeros(hyperparam(6),hyperparam(2));
del_b_out = 0;
del_w_hidden = 0;
del_b_hidden = 0;
del_embed = 0;

for k=1:length(batch)
    sample_word = batch(:,k);
    desired_output = zeros(hyperparam(2),1);
    desired_output(batch_output(k)) = 1;
    
    one_hot_first = zeros(hyperparam(2),1);
    one_hot_first(sample_word(1),1) = 1;
    one_hot_second = zeros(hyperparam(2),1);
    one_hot_second(sample_word(2),1) = 1;
    one_hot_third = zeros(hyperparam(2),1);
    one_hot_third(sample_word(3),1) = 1;
    
    pre_word = embed'*[one_hot_first, one_hot_second, one_hot_third];
    sample_word = reshape(pre_word, 3*hyperparam(7), 1);
    
    v_hidden = w_hidden' * sample_word + b_hidden;
    o_hidden = ones(hyperparam(6),1)./(1+exp(-v_hidden));  
    v_output = w_output' * o_hidden + b_output;
    o_output = exp(v_output)./sum(exp(v_output));
    
    local_gradient_output = desired_output - o_output;
    delta_w_output = -( o_hidden * local_gradient_output' );
    delta_b_output = -( local_gradient_output);

    error_hidden = w_output * local_gradient_output;
    local_gradient_hidden = error_hidden .* o_hidden .* (1-o_hidden);
    delta_w_hidden = -(sample_word* local_gradient_hidden');
    delta_b_hidden = -(local_gradient_hidden)* ones(size(v_hidden,2), 1);
    
    error_embed = w_hidden * local_gradient_hidden; 
    
    error_embed = reshape(error_embed, hyperparam(7), 3);
    
    del_embed_batch = -(one_hot_first*error_embed(:,1)' + one_hot_second*error_embed(:,2)' + one_hot_third*error_embed(:,3)');
    
    training_error = training_error + abs(sum(desired_output.*log(o_output)) );
    
    del_w_out =  del_w_out + delta_w_output;
    del_b_out = del_b_out +delta_b_output;
    del_w_hidden = del_w_hidden+delta_w_hidden;
    del_b_hidden = del_b_hidden +delta_b_hidden;
    del_embed = del_embed  + del_embed_batch;    
    
end

end

function [rate] = triagram_list(w_output, b_output, w_hidden, b_hidden, embed, hyperparam, testx, testd, sample_index_table, words)

for i = 1:length(sample_index_table )
    sample_for_test = testx(:,sample_index_table(i));
    label_for_test = testd(sample_index_table(i),:);
    prob = strings(10,2);
    word_1 = words(sample_for_test(1));
    word_2 = words(sample_for_test(2));
    word_3 = words(sample_for_test(3));
    
    % apply one hot encoding
    one_hot1=zeros(250,1);
    one_hot1(sample_for_test(1),1)=1;
    one_hot2=zeros(250,1);
    one_hot2(sample_for_test(2),1)=1;
    one_hot3=zeros(250,1);
    one_hot3(sample_for_test(3),1)=1;
    
    % embeding matrix stage
    sample_for_test=[embed'*one_hot1 ; embed'*one_hot2; embed'*one_hot3];
    
    % forward pass stage
    v_hidden=w_hidden'*sample_for_test+b_hidden;
    o_hidden=ones(hyperparam(6),1)./(1+exp(-v_hidden));
    v_output=w_output'*o_hidden+b_output;
    o_output=exp(v_output)./sum(exp(v_output));
    [values,indices] = maxk(o_output, 10);
    
    prob(:,1)=words(indices);
    prob(:,2)=100*values;
    disp("Trigram : "  + word_1  + word_2  +  word_3);
    disp("Desired 4th Word: " + words(label_for_test))
    disp("10 Words with Maximum Probabilities : ");
    disp(prob);
    disp(' ');
end
rate = 0;

for k=1:length(testx)
    sample_for_test = testx(:,k);
    
    one_hot1=zeros(250,1);
    one_hot1(sample_for_test(1),1)=1;
    one_hot2=zeros(250,1);
    one_hot2(sample_for_test(2),1)=1;
    one_hot3=zeros(250,1);
    one_hot3(sample_for_test(3),1)=1;
    
    sample_for_test=[embed'*one_hot1 ; embed'*one_hot2; embed'*one_hot3];

    v_hidden=w_hidden'*sample_for_test+b_hidden;
    o_hidden=ones(hyperparam(6),1)./(1+exp(-v_hidden));
    v_output=w_output'*o_hidden+b_output;
    o_output=exp(v_output)./sum(exp(v_output));
    [values,indices] = maxk(o_output, 1);
    
    if ((indices == testd(k)))
        rate = rate+1;
    end
end
rate = rate/length(testx)*100;
end

