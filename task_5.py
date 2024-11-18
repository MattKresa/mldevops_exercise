import optuna

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    epochs = trial.suggest_int("epochs", 5, 30)
    
    model = FashionMNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images)

            # Compute loss
            loss = criterion(output, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if accuracy > 95:  # Stop if accuracy exceeds 95%
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return accuracy  # Return the final accuracy for the trial

study = optuna.create_study(direction="maximize")  # Maximize accuracy
study.optimize(objective, n_trials=5)  # Run 5 trials

# Print the best parameters
print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")