
import torch.nn as nn
# client update
criterion = nn.CrossEntropyLoss()


def client_update(model, optimizer, data_stream, train_losses, train_accs):
    model.train()
    correct = 0
    data, target = data_stream
    target = target.long()
    # data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    train_losses.append(loss.item())
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    train_accs.append(correct / len(data))
    loss.backward()
    optimizer.step()


