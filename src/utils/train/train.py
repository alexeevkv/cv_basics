import torch
from .tracker import NetTracker


def get_weights_and_grads(named_parameters):
    avg_weights = list()
    avg_grads = list()

    for _, parameter in named_parameters:
        mean_weight = parameter.detach().cpu().abs().mean()
        mean_grad = parameter.grad.detach().abs().mean()

        avg_weights.append(mean_weight)
        avg_grads.append(mean_grad)

    avg_weight = sum(avg_weights) / len(avg_weights)
    avg_grad = sum(avg_grads) / len(avg_grads)

    avg_weight = float(avg_weight.cpu().numpy())
    avg_grad = float(avg_grad.cpu().numpy())

    return avg_weight, avg_grad


def train(
    model,
    tracker: NetTracker,
    epoch_num,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
):

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

        # get weight and grads after each training epoch

        avg_weight, avg_grad = get_weights_and_grads(model.named_parameters())

        tracker.track_weights(avg_weight, avg_grad)

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
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}"
        )
        print(
            f"Epoch {epoch + 1} \t\t Training accuracy: {train_correct / len(train_loader.dataset)}"
            f"\t\t Validation accuracy: {val_correct / len(val_loader.dataset)}"
        )
        tracker.track_losses(train_loss, val_loss)

        if epoch % 5 == 0:
            print(f"It is {epoch} so model weights wil be saved into NetTracker")
            tracker.save_weights(model, epoch)
