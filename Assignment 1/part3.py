import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.optim as optim

# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)


def create_mlp(input_size, hidden_layers, num_neurons, output_size, activation):
    layers = []
    layers.append(nn.Flatten())
    for _ in range(hidden_layers):
        layers.append(nn.Linear(input_size, num_neurons))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        input_size = num_neurons
    layers.append(nn.Linear(input_size, output_size))

    return nn.Sequential(*layers)

#train every hyperparameter combination
def train_model(model, optimizer, criterion, train_data, train_labels, val_data, val_labels, test_data, test_labels, max_epochs, num_runs, patience):
    valid_accuracy_per_run = []
    test_accuracy_per_run = []

    # for run in range(num_runs):
    best_valid_accuracy = 0
    epochs_without_improvement = 0
    model.train()
    for epoch in range(max_epochs):
            # Training loop on the combined dataset
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = torch.sum(torch.argmax(outputs, dim=1) == train_labels).item() / len(train_labels) * 100

            # Validation loop
        model.eval()
        with torch.no_grad():
            valid_outputs = model(val_data)  # Use combined data for validation
            valid_loss = criterion(valid_outputs, val_labels)
            valid_accuracy = torch.sum(torch.argmax(valid_outputs, dim=1) == val_labels).item() / len(val_labels) * 100
            valid_accuracy_per_run.append(valid_accuracy)

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement+1

            if epochs_without_improvement >= patience:
                break

        print(" Epoch %d - Train Loss %.4f - Train Accuracy %.2f - Validation Loss %.4f - Validation Accuracy %.2f" % (epoch+1, loss.item(), train_acc, valid_loss.item(), valid_accuracy))

        # Test loop
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_accuracy = torch.sum(torch.argmax(test_outputs, dim=1) == test_labels).item() / len(test_labels) * 100
        test_accuracy_per_run.append(test_accuracy)

    return valid_accuracy_per_run, test_accuracy_per_run

#training on combined dataset after finding the best model
def train_model_combined_test(model, optimizer, criterion, combined_data, combined_labels, test_data, test_labels, max_epochs, num_runs, patience):
    valid_accuracy_per_run = []
    test_acc_per_run = []

    
    best_valid_accuracy = 0
    epochs_without_improvement = 0
    model.train()
    for epoch in range(max_epochs):
        
        optimizer.zero_grad()
        outputs = model(combined_data)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_accuracy = torch.sum(torch.argmax(outputs, dim=1) == combined_labels).item() / len(combined_labels) * 100

        model.eval()
        with torch.no_grad():
            valid_outputs = model(combined_data)
            valid_loss = criterion(valid_outputs, combined_labels)
            valid_accuracy = torch.sum(torch.argmax(valid_outputs, dim=1) == combined_labels).item() / len(combined_labels) * 100
            valid_accuracy_per_run.append(valid_accuracy)

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = epochs_without_improvement + 1

            if epochs_without_improvement >= patience:
                break

        print(" Epoch %d - Train Loss %.4f - Train Accuracy %.2f - Validation Loss %.4f - Validation Accuracy %.2f" % ( epoch + 1, loss.item(), train_accuracy, valid_loss.item(), valid_accuracy))

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_acc = torch.sum(torch.argmax(test_outputs, dim=1) == test_labels).item() / len(test_labels) * 100
        test_acc_per_run.append(test_acc)

    return valid_accuracy_per_run, test_acc_per_run

#grid search
def hyperparameter_tuning(train_data, train_labels, val_data, val_labels, test_data, test_labels, hidden_layers_values, num_neurons, learning_rate_values, activation_functions, max_epochs_values, batch_sizes, patience, num_runs):
    best_accuracy = -1
    best_hyperparameters = None
    

    for hidden_layers in hidden_layers_values:
        for neurons in num_neurons:
            for learning_rate in learning_rate_values:
                for activation in activation_functions:
                    for max_epochs in max_epochs_values:
                        for batch_size in batch_sizes:
                            val_acc_per_config = []
                            test_acc_per_config = []

                            for _ in range(num_runs):
                                input_size = 784
                                output_size = 10
                                model = create_mlp(input_size, hidden_layers, neurons, output_size, activation)
                                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                                criterion = nn.CrossEntropyLoss()
                                
                                val_accuracy, test_accuracy = train_model(model, optimizer, criterion, train_data, train_labels, val_data, val_labels, test_data, test_labels, max_epochs, num_runs, patience)
                                val_acc_per_config.extend(val_accuracy)
                                test_acc_per_config.extend(test_accuracy)
                            
                            mean_val_acc, margin_of_error = confidence_interval(val_acc_per_config)
                            print(f"Hyperparameters: Hidden Layers={hidden_layers}, Number of Neurons={neurons}, Learning Rate={learning_rate}, Activation={activation}, Max Epochs={max_epochs}, Batch Size={batch_size}")
                            print("Validation Accuracy Confidence Interval:", mean_val_acc, "+-",margin_of_error)
                            
                            if mean_val_acc > best_accuracy:
                                best_accuracy = mean_val_acc
                                best_hyperparameters = {
                                    'hidden_layers': hidden_layers,
                                    'num_neurons': neurons,
                                    'learning_rate': learning_rate,
                                    'activation': activation,
                                    'max_epochs': max_epochs,
                                    'batch_size': batch_size
                                }

    return best_hyperparameters, best_accuracy

#confidence interval calculation
def confidence_interval(data):
    n = len(data)
    mean_value = np.mean(data)
    stddev = np.std(data)
    margin_error = 1.96 * (stddev / np.sqrt(n))
    return mean_value, margin_error

#hyperparameters
hidden_layers_values = [1,2]
num_neurons = [10, 30]
learning_rate_values = [0.01,0.1,0.001,0.0001]
activation_functions = ['sigmoid','tanh','relu']
batch_sizes = [32]
max_epochs_values = [50]
patience = 5 #early stopping

#num of runs for each combination
num_runs = 10

#grid search and best hyperparameters
best_hyperparameters, best_accuracy = hyperparameter_tuning(x_train, y_train, x_validation, y_validation, x_test, y_test, hidden_layers_values, num_neurons, learning_rate_values, activation_functions, max_epochs_values, batch_sizes, patience, num_runs)

#combining the dataset after grid search
x_combined = torch.cat((x_train, x_validation), dim=0)
y_combined = torch.cat((y_train, y_validation), dim=0)

#training the model on best parameters 10 times
test_acc_per_run = []
for _ in range(num_runs):
    input_size = 784
    output_size = 10
    model = create_mlp(input_size,  best_hyperparameters['hidden_layers'], best_hyperparameters['num_neurons'], output_size, best_hyperparameters['activation'])
    optimizer = optim.Adam(model.parameters(), lr=best_hyperparameters['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    val_acc_per_run, test_accuracy = train_model_combined_test(model, optimizer, criterion, x_combined, y_combined, x_test, y_test, best_hyperparameters['max_epochs'], num_runs, patience)
    test_acc_per_run.append(test_accuracy)
    
test_means_and_errors = [confidence_interval(test_acc_per_run)]
mean, error = test_means_and_errors[0]

#printing the test results
print("Best Hyperparameters:", best_hyperparameters)
print("Test Accuracy Confidence Interval: =", mean," +- ", error)