#Train

import torch
import torch.optim as optim
from tqdm import tqdm
from config import batchsize, epochs, learning_rate, batch_norm_momentum, batch_norm_eps, epsilon, device, data_dir, csv_path, weights_dir

def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}, Training Loss: {avg_loss:.4f}")