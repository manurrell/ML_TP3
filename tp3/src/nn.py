import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers, output_dim, learning_rate=0.01, rs="", min_lr= 0.01, rsr = 0.99, batch_size = 0, adam= 0, beta1=0.9, beta2=0.9999, e=1e-8, lambda_reg=0.01, patience=50, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epoch = 0
        self.lambda_reg = lambda_reg
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e
        self.t = 0
        self.batch_size = batch_size
        self.scheduler = rs 
        self.min_lr = min_lr
        self.decay_rate = rsr
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.layers = [input_dim] + hidden_layers + [output_dim]
        self.L = len(self.layers) - 1  
        
        
        self.weights = []
        self.biases = []
        for i in range(self.L):
            W = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i])# He initialization escalado para que no de problemas
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        if adam:
            self.m_w = [np.zeros_like(W) for W in self.weights]
            self.v_w = [np.zeros_like(W) for W in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss= np.sum(log_likelihood) / m
        if self.lambda_reg > 0:
            l2_loss = 0
            for W in self.weights:
                l2_loss += np.sum(W ** 2)
            loss += (self.lambda_reg / (2 * m)) * l2_loss  # l2
        return loss

    def forward(self, X):
        activations = [X]
        pre_activations = []

        A = X
        for i in range(self.L - 1):
            Z = A @ self.weights[i] + self.biases[i]
            A = self.relu(Z)
            pre_activations.append(Z)
            activations.append(A)
        
        # salida softmax
        Z = A @ self.weights[-1] + self.biases[-1]
        A = self.softmax(Z)
        pre_activations.append(Z)
        activations.append(A)
        
        return activations, pre_activations

    def backward(self, X, y, activations, pre_activations):
        m = X.shape[0]
        grads_W = [0] * self.L
        grads_b = [0] * self.L

        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y] = 1

        # grad en la capa de salida
        dZ = activations[-1] - y_onehot
        grads_W[-1] = activations[-2].T @ dZ / m
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True) / m

        #back
        for l in reversed(range(self.L - 1)):
            dA = dZ @ self.weights[l + 1].T
            dZ = dA * self.relu_derivative(pre_activations[l])
            grads_W[l] = activations[l].T @ dZ / m
            grads_b[l] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # actualizar parametros
        for l in range(self.L):
            if self.adam:
                self.t += 1
                self.m_w[l] = self.beta1 * self.m_w[l] + (1 - self.beta1) * grads_W[l]
                self.v_w[l] = self.beta2 * self.v_w[l] + (1 - self.beta2) * (grads_W[l] ** 2)
                
                m_w_hat = self.m_w[l] / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w[l] / (1 - self.beta2 ** self.t)


                self.weights[l] -= self.learning_rate * (m_w_hat / (np.sqrt(v_w_hat) + self.e) + self.lambda_reg * self.weights[l])

                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * grads_b[l]
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (grads_b[l] ** 2)

                m_b_hat = self.m_b[l] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[l] / (1 - self.beta2 ** self.t)

                self.biases[l] -= self.learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + self.e))
            else:

                self.weights[l] -= self.learning_rate * (grads_W[l] + self.lambda_reg * self.weights[l])
                self.biases[l] -= self.learning_rate * grads_b[l]


    def fit(self, X, y, X_val, y_val, epochs=1000 ):
        train_accuracies = []
        val_accuracies = []
        for epoch in range(epochs):
            if self.scheduler == "linear":
                self.learning_rate = max(self.min_lr, self.initial_lr * (1 - epoch / epochs))
            elif self.scheduler == "exponential":
                self.learning_rate = max(self.min_lr,self.initial_lr * (self.decay_rate ** epoch))
            if self.batch_size:
                perm = np.random.permutation(X.shape[0])
                X = X[perm]
                y = y[perm]
                for i in range(0, X.shape[0], self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    activations, pre_activations = self.forward(X_batch)
                    loss = self.cross_entropy_loss(activations[-1], y_batch)
                    self.backward(X_batch, y_batch, activations, pre_activations)

            else:
                activations, pre_activations = self.forward(X)
                loss = self.cross_entropy_loss(activations[-1], y)
                self.backward(X, y, activations, pre_activations)
                
            val_activations, _ = self.forward(X_val)       
            val_loss = self.cross_entropy_loss(val_activations[-1], y_val)

            train_acc = self.accuracy(X, y)
            val_acc = self.accuracy(X_val, y_val)
            train_accuracies.append((epoch, train_acc))
            val_accuracies.append((epoch, val_acc))

            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.bad_epoch = 0
            else:
                self.bad_epoch += 1


            if self.bad_epoch >= self.patience:
                print(f"Early stopping {epoch}")
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {train_acc:.4f}")
                break

            if epoch % 100 == 0 or epoch == epochs -1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {train_acc:.4f}")
        return train_accuracies, val_accuracies

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        num_classes = self.layers[-1]
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, y_pred):
            cm[true, pred] += 1
        return cm
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(25, 25)) 
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        
        # Etiquetas de los números en cada celda
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        
    def evaluate(self, X, y, cm_show=1):
        activations, _ = self.forward(X)
        loss = self.cross_entropy_loss(activations[-1], y)
        acc = self.accuracy(X, y)
        cm = self.confusion_matrix(X, y)
        print(f"Accuracy: {acc:.4f}")
        print(f"Cross-Entropy Loss: {loss:.4f}")
        if cm_show:
                
            print("Matriz de Confusión:")
            self.plot_confusion_matrix(cm)
class TorchNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

    


def plot_loss_and_accuracy(train_acc, val_acc):
    epochs_train, acc_train = zip(*train_acc)
    epochs_val, acc_val = zip(*val_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train, acc_train, label='Train Accuracy', marker='o')
    plt.plot(epochs_val, acc_val, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train(model, X_train, y_train, X_val, y_val, l2=0.01, p = 10, adam=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # ds
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=1250, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1250)
    
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=l2)
        
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_acc_list = []
    val_acc_list = []
    
    for epoch in range(500): 
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * len(xb)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)
        val_loss /= len(val_dataset)
        
        if p == 0:
            best_model_state = model.state_dict()
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
        model.eval()
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == yb).sum().item()
                total_train += yb.size(0)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        train_acc_list.append((epoch, train_acc))
        val_acc_list.append((epoch, val_acc))

        if p!= 0:
            if patience_counter >= p:
                print(f"Early stopping at epoch {epoch}")
                break

    
    model.load_state_dict(best_model_state)
    return model, train_acc_list, val_acc_list
def evaluate(model, X, y, batch_size=1250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.item() * xb.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

