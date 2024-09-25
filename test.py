import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from fedpredict import fedpredict_client_torch
# test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = test_loader
        target = target.long()
        # data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target = target.cpu()
        # pred = pred.cpu()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    # print(f'\n\t\tTest: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data)} F1-score: {test_f1:.4f}'
    #       f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy, test_f1


def test1(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    modified_dataset = []  # save date after predict

    with torch.no_grad():
        data, target = test_loader
        # print(f"data shape: {data.shape}, Target shape: {target.shape}")
        target = target.long()
        # data, target = data.to(device), target.to(device)
        output = model(data)
        # print(f"Output shape: {output.shape}")
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        is_correct = 1- pred.eq(target.view_as(pred)).float()  # correct 1，false 0  为了ddm后续修改为correct 0，false 1
        modified_dataset.append((data, is_correct))
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target = target.cpu()
        # pred = pred.cpu()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    # print(f'\n\t\tTest: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data)} F1-score: {test_f1:.4f}'
    #       f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy, test_f1, modified_dataset


def test2(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    modified_dataset = []  # save date after predict
    with torch.no_grad():
        data, target = test_loader
        target = target.long()
        output = model(data)
        # Calculate CrossEntropyLoss
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        # Apply Softmax to get class probabilities
        softmax_output = F.softmax(output, dim=1)
        # Get predicted class index
        pred = output.argmax(dim=1, keepdim=True)

        # Calculate confidence as the probability of the predicted class
        confidence = softmax_output.gather(1, pred)

        # Modify is_correct based on confidence
        is_correct = confidence  # Use probability as confidence

        # Append data, confidence, and is_correct to modified_dataset
        modified_dataset.append((data, is_correct))

        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)

    # Note: You need to handle F1-score calculation appropriately based on your requirements
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')

    return test_loss, test_accuracy, test_f1, modified_dataset

def testfedpredict(model, global_model, test_loader,t,T,nt):
    model.eval()
    test_loss = 0
    correct = 0
    modified_dataset = []  # save date after predict

    with torch.no_grad():
        data, target = test_loader
        # print(f"data shape: {data.shape}, Target shape: {target.shape}")
        target = target.long()
        # data, target = data.to(device), target.to(device)
        combinel_model = fedpredict_client_torch(local_model=model,
                                                 global_model=global_model,
                                                 t=t,
                                                 T=T,
                                                 nt=nt,dynamic=True)
        # Use the combined model to perform predictions over the input data
        output = combinel_model(data)
        # print(f"Output shape: {output.shape}")
        test_loss += nn.CrossEntropyLoss()(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        is_correct = 1- pred.eq(target.view_as(pred)).float()  # correct 1，false 0  为了ddm后续修改为correct 0，false 1
        modified_dataset.append((data, is_correct))
        correct += pred.eq(target.view_as(pred)).sum().item()
        # target = target.cpu()
        # pred = pred.cpu()

    test_loss /= len(data)
    test_accuracy = 100. * correct / len(data)
    test_f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average='weighted')
    # print(f'\n\t\tTest: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data)} F1-score: {test_f1:.4f}'
    #       f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy, test_f1, modified_dataset