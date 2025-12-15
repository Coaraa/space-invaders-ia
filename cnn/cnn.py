import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

class CoordinateCNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(CoordinateCNN, self).__init__()
        
        # Convolution 1D : Traite le vecteur comme une séquence temporelle ou spatiale simple
        # Entrée : [Batch, 1, 6] (1 canal, 6 caractéristiques)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32 * state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        # Il faut redimensionner l'entrée pour le CNN : [Batch, Channels, Length]
        # x arrive en [Batch, 6] -> on le passe en [Batch, 1, 6]
        x = x.unsqueeze(1) 
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Aplatir pour les couches denses
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x) # Pas de softmax ici, c'est inclus dans la loss CrossEntropy
    
    
def train_supervised(model, X, y, epochs=20, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Création du DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss() # Pour la classification multi-classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("--- Début de l'entraînement supervisé ---")
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
    return model, loss_history

