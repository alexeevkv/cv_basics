import torch


def train(
    model,
    epoch_num,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    evaluate_training=True,
):
    train_losses, val_losses = list(), list()

    for epoch in range(epoch_num):
        # train
        model.train()
        train_loss = 0
        train_correct = 0
        for data in train_loader:
            inputs, targets = data['image'], data['target']
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(outputs, dim=1)
            train_correct += torch.sum(pred == targets).item()

        # eval
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data['image'], data['target']
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, pred = torch.max(outputs, dim=1)
                val_correct += torch.sum(pred == targets).item()

        # print train_loss and val_loss on i epoch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}"
        )
        print(
            f"Epoch {epoch + 1} \t\t Training accuracy: {train_correct / len(train_loader.dataset)}"
            f"\t\t Validation accuracy: {val_correct / len(val_loader.dataset)}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    if evaluate_training:
        return model, train_losses, val_losses
    else:
        return model
