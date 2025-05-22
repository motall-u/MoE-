from models import MoE ,  Expert , Gating
import torch
import torch.nn as nn
import torch.optim as optim 





def main():
    # Generate the dataset
    num_samples = 5000
    input_dim = 4
    hidden_dim = 32

    # Generate equal numbers of labels 0, 1, and 2
    y_data = torch.cat([
        torch.zeros(num_samples // 3),
        torch.ones(num_samples // 3),
        torch.full((num_samples - 2 * (num_samples // 3),), 2)  # Filling the remaining to ensure exact num_samples
    ]).long()

    # Biasing the data based on the labels
    x_data = torch.randn(num_samples, input_dim)

    for i in range(num_samples):
        if y_data[i] == 0:
            x_data[i, 0] += 1  # Making x[0] more positive
        elif y_data[i] == 1:
            x_data[i, 1] -= 1  # Making x[1] more negative
        elif y_data[i] == 2:
            x_data[i, 0] -= 1  # Making x[0] more negative

    # Shuffle the data to randomize the order
    indices = torch.randperm(num_samples)
    x_data = x_data[indices]
    y_data = y_data[indices]

    # Verify the label distribution
    y_data.bincount()

    # Shuffle the data to ensure x_data and y_data remain aligned
    shuffled_indices = torch.randperm(num_samples)
    x_data = x_data[shuffled_indices]
    y_data = y_data[shuffled_indices]

    # Splitting data for training individual experts
    # Use the first half samples for training individual experts
    x_train_experts_full = x_data[:int(num_samples//2)]
    y_train_experts_full = y_data[:int(num_samples//2)]

    # Define masks for expert-specific data based on y_train_experts_full
    mask_expert1_full = (y_train_experts_full == 0) | (y_train_experts_full == 1)
    mask_expert2_full = (y_train_experts_full == 1) | (y_train_experts_full == 2)
    mask_expert3_full = (y_train_experts_full == 0) | (y_train_experts_full == 2)

    # Determine the number of samples for each expert's specific dataset before splitting
    num_samples_expert1_full = mask_expert1_full.sum().item()
    num_samples_expert2_full = mask_expert2_full.sum().item()
    num_samples_expert3_full = mask_expert3_full.sum().item()
    
    # To ensure all experts have a comparable number of samples for training AND validation
    # We will base the num_samples_per_expert on the smallest pool after masking
    num_samples_per_expert_pool = min(num_samples_expert1_full, num_samples_expert2_full, num_samples_expert3_full)

    # Data for Expert 1
    x_expert1_full = x_train_experts_full[mask_expert1_full][:num_samples_per_expert_pool]
    y_expert1_full = y_train_experts_full[mask_expert1_full][:num_samples_per_expert_pool]
    expert1_train_split = int(0.8 * len(x_expert1_full))
    x_expert1_train = x_expert1_full[:expert1_train_split]
    y_expert1_train = y_expert1_full[:expert1_train_split]
    x_expert1_val = x_expert1_full[expert1_train_split:]
    y_expert1_val = y_expert1_full[expert1_train_split:]

    # Data for Expert 2
    x_expert2_full = x_train_experts_full[mask_expert2_full][:num_samples_per_expert_pool]
    y_expert2_full = y_train_experts_full[mask_expert2_full][:num_samples_per_expert_pool]
    expert2_train_split = int(0.8 * len(x_expert2_full))
    x_expert2_train = x_expert2_full[:expert2_train_split]
    y_expert2_train = y_expert2_full[:expert2_train_split]
    x_expert2_val = x_expert2_full[expert2_train_split:]
    y_expert2_val = y_expert2_full[expert2_train_split:]

    # Data for Expert 3
    x_expert3_full = x_train_experts_full[mask_expert3_full][:num_samples_per_expert_pool]
    y_expert3_full = y_train_experts_full[mask_expert3_full][:num_samples_per_expert_pool]
    expert3_train_split = int(0.8 * len(x_expert3_full))
    x_expert3_train = x_expert3_full[:expert3_train_split]
    y_expert3_train = y_expert3_full[:expert3_train_split]
    x_expert3_val = x_expert3_full[expert3_train_split:]
    y_expert3_val = y_expert3_full[expert3_train_split:]

    # Splitting the next half samples for training MoE model, validation, and for testing
    x_remaining = x_data[num_samples//2:] # Corrected index
    y_remaining = y_data[num_samples//2:] # Corrected index

    # Ensure num_samples is even, or adjust indices if not, to avoid off-by-one from previous slice
    # The current num_samples = 5000 is even, so int(num_samples/2) is fine.
    # If num_samples was odd, [int(num_samples/2)+1:] would skip a sample.
    # Using [num_samples//2:] is safer for the second half.

    train_val_split = int(0.8 * len(x_remaining)) # 80% for combined train+val for MoE
    x_train_val_moe = x_remaining[:train_val_split]
    y_train_val_moe = y_remaining[:train_val_split]
    
    x_test = x_remaining[train_val_split:]
    y_test = y_remaining[train_val_split:]

    # Further split x_train_val_moe into train and validation for MoE
    # This means MoE train is 80% of 80% = 64% of x_remaining
    # MoE val is 20% of 80% = 16% of x_remaining
    # Test is 20% of x_remaining.
    # This is not exactly 80/10/10 of x_remaining. Let's adjust for 80/10/10 of x_remaining.
    
    num_remaining = len(x_remaining)
    moe_train_end_idx = int(0.8 * num_remaining)
    moe_val_end_idx = int(0.9 * num_remaining)

    x_train_moe = x_remaining[:moe_train_end_idx]
    y_train_moe = y_remaining[:moe_train_end_idx]

    x_val_moe = x_remaining[moe_train_end_idx:moe_val_end_idx]
    y_val_moe = y_remaining[moe_train_end_idx:moe_val_end_idx]

    x_test = x_remaining[moe_val_end_idx:] # Test is the final 10%
    y_test = y_remaining[moe_val_end_idx:]


    print("MoE training data shape:", x_train_moe.shape)
    print("MoE validation data shape:", x_val_moe.shape)
    print("Test data shape:", x_test.shape)
    print("Expert 1 training data shape:", x_expert1_train.shape)
    print("Expert 1 validation data shape:", x_expert1_val.shape)
    print("Expert 2 training data shape:", x_expert2_train.shape)
    print("Expert 2 validation data shape:", x_expert2_val.shape)
    print("Expert 3 training data shape:", x_expert3_train.shape)
    print("Expert 3 validation data shape:", x_expert3_val.shape)


    # Define hidden dimension
    output_dim = 3
    hidden_dim = 32

    epochs = 500
    learning_rate = 0.001
    batch_size = 32 # Added batch_size

    from torch.utils.data import TensorDataset, DataLoader # Added imports
    from sklearn.metrics import precision_score, recall_score, f1_score # Added for more metrics

    # Define the evaluation function early so it can be used in training loops
    def evaluate(model, data_loader): # Modified to accept DataLoader
        model.eval() # Ensure model is in evaluation mode
        all_labels = []
        all_predicted = []
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                _, predicted_batch = torch.max(outputs, 1) # Renamed to avoid clash
                
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted_batch.cpu().numpy()) # Use renamed variable
                
                total_correct += (predicted_batch == labels).sum().item() # Use renamed variable
                total_samples += labels.size(0)
        
        if total_samples == 0: 
            return 0.0, 0.0, 0.0, 0.0 # accuracy, precision, recall, f1

        accuracy = total_correct / total_samples
        # Use zero_division=0 to prevent warnings/errors if a class is not predicted
        # Using macro average for multi-class metrics
        precision = precision_score(all_labels, all_predicted, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predicted, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predicted, average='macro', zero_division=0)
        
        return accuracy, precision, recall, f1

    # Instantiate the experts
    expert1 = Expert(input_dim, hidden_dim, output_dim)
    expert2 = Expert(input_dim, hidden_dim, output_dim)
    expert3 = Expert(input_dim, hidden_dim, output_dim)

    # Set up loss
    criterion = nn.CrossEntropyLoss()

    # Optimizers for experts
    optimizer_expert1 = optim.Adam(expert1.parameters(), lr=learning_rate)
    optimizer_expert2 = optim.Adam(expert2.parameters(), lr=learning_rate)
    optimizer_expert3 = optim.Adam(expert3.parameters(), lr=learning_rate)

    # Training loop for expert 1
    for epoch in range(epochs):
        expert1.train()
        for batch_x, batch_y in train_loader_e1: # Iterate over DataLoader
            optimizer_expert1.zero_grad()
            outputs_expert1 = expert1(batch_x)
            loss_expert1 = criterion(outputs_expert1, batch_y)
            loss_expert1.backward()
            optimizer_expert1.step()
        if (epoch + 1) % 100 == 0: # Evaluate on validation set every 100 epochs
            acc, prec, rec, f1_val = evaluate(expert1, val_loader_e1) # Use validation DataLoader
            print(f"Expert 1 - Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1_val:.4f}")

    # Training loop for expert 2
    for epoch in range(epochs):
        expert2.train()
        for batch_x, batch_y in train_loader_e2: # Iterate over DataLoader
            optimizer_expert2.zero_grad()
            outputs_expert2 = expert2(batch_x)
            loss_expert2 = criterion(outputs_expert2, batch_y)
            loss_expert2.backward()
            optimizer_expert2.step()
        if (epoch + 1) % 100 == 0: # Evaluate on validation set every 100 epochs
            acc, prec, rec, f1_val = evaluate(expert2, val_loader_e2) # Use validation DataLoader
            print(f"Expert 2 - Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1_val:.4f}")

    # Training loop for expert 3
    for epoch in range(epochs):
        expert3.train()
        for batch_x, batch_y in train_loader_e3: # Iterate over DataLoader
            optimizer_expert3.zero_grad()
            outputs_expert3 = expert3(batch_x)
            loss_expert3 = criterion(outputs_expert3, batch_y)
            loss_expert3.backward()
            optimizer_expert3.step()
        if (epoch + 1) % 100 == 0: # Evaluate on validation set every 100 epochs
            acc, prec, rec, f1_val = evaluate(expert3, val_loader_e3) # Use validation DataLoader
            print(f"Expert 3 - Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1_val:.4f}")
    
    # Create the MoE model with the trained experts
    moe_model = MoE([expert1, expert2, expert3])

    # Train the MoE model
    optimizer_moe = optim.Adam(moe_model.gating.parameters(), lr=learning_rate) # Only train gating network
    for epoch in range(epochs):
        moe_model.train() # Sets gating to train mode, experts are frozen
        for batch_x, batch_y in train_loader_moe: # Iterate over DataLoader
            optimizer_moe.zero_grad()
            outputs_moe = moe_model(batch_x)
            loss_moe = criterion(outputs_moe, batch_y)
            loss_moe.backward()
            optimizer_moe.step()
        if (epoch + 1) % 100 == 0: # Evaluate on validation set every 100 epochs
            acc, prec, rec, f1_val = evaluate(moe_model, val_loader_moe) # Use validation DataLoader
            print(f"MoE - Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1_val:.4f}")

    # Final evaluation on Test Set
    print("\n--- Final Test Set Evaluation ---")
    acc_e1, prec_e1, rec_e1, f1_e1 = evaluate(expert1, test_loader)
    print(f"Expert 1 Test Metrics - Accuracy: {acc_e1:.4f}, Precision: {prec_e1:.4f}, Recall: {rec_e1:.4f}, F1-Score: {f1_e1:.4f}")

    acc_e2, prec_e2, rec_e2, f1_e2 = evaluate(expert2, test_loader)
    print(f"Expert 2 Test Metrics - Accuracy: {acc_e2:.4f}, Precision: {prec_e2:.4f}, Recall: {rec_e2:.4f}, F1-Score: {f1_e2:.4f}")

    acc_e3, prec_e3, rec_e3, f1_e3 = evaluate(expert3, test_loader)
    print(f"Expert 3 Test Metrics - Accuracy: {acc_e3:.4f}, Precision: {prec_e3:.4f}, Recall: {rec_e3:.4f}, F1-Score: {f1_e3:.4f}")

    acc_moe, prec_moe, rec_moe, f1_moe = evaluate(moe_model, test_loader)
    print(f"MoE Test Metrics - Accuracy: {acc_moe:.4f}, Precision: {prec_moe:.4f}, Recall: {rec_moe:.4f}, F1-Score: {f1_moe:.4f}")


if __name__ == "__main__":
    main()