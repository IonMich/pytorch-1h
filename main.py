import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct / total_examples).item()


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(123)
    X_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )

    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor(
        [
            [-0.8, 2.8],
            [2.6, -1.6],
        ]
    )
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)

    train_loader = DataLoader(
        dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
    )

    test_ds = ToyDataset(X_test, y_test)

    test_loader = DataLoader(
        dataset=test_ds, batch_size=2, shuffle=False, num_workers=0, drop_last=False
    )

    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx + 1}:", x, y)


    import torch.nn.functional as F

    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    with torch.no_grad():
        out = torch.softmax(model(X_test.to(device)), dim=1)
    print(out)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params)

    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)

            loss = F.cross_entropy(logits, labels)  # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### LOGGING
            print(
                f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train/Val Loss: {loss:.2f}"
            )

        model.eval()
        train_acc = compute_accuracy(model, train_loader)
        test_acc = compute_accuracy(model, test_loader)
        print(
            f"Epoch: {epoch + 1:03d}/{num_epochs:03d}"
            f" | Train/Val Acc: {train_acc:.2f}/{test_acc:.2f}"
        )
    print("Training complete.")
