import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0

        correct += torch.sum(pred == target).item()
    return correct / len(target)


def mixup_accuracy(output, targets_a, targets_b, lam):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        
        correct = 0
        
        correct += (lam * pred.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * pred.eq(targets_b.data).cpu().sum().float())
    
    return correct / len(targets_a)
   

# def top_k_acc(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

def binarize(output):
    pred = torch.round(torch.sigmoid(output))
    return pred

# def accuracy(output, target):
#     with torch.no_grad():
#         pred = binarize(output)
#         correct = 0
#         correct += torch.sum(pred == target).item() / len(target[1])
#     return correct / len(target)

def label_wise_accuracy(output, target):
    with torch.no_grad():
        pred = binarize(output)
        assert pred.shape[0] == len(target)
        label_wise_acc = list()
        for i in range(pred.shape[1]):
            correct = 0
            correct += torch.sum(pred[:, i] == target[:, i])
            label_wise_acc.append(correct)
    return label_wise_acc / len(target)

def confusion_metrics(output, target):
    with torch.no_grad():
        pred = binarize(output)
        assert pred.shape[0] == len(target)
        TP = torch.sum(pred * target).item()
        TN = torch.sum((1-pred) * (1-target)).item()
        FP = torch.sum(pred * (1-target)).item()
        FN = torch.sum((1-pred) * target).item()
    return TP, FP, TN, FN

def truepositive(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TP = torch.sum(pred * target).item()
    return TP / len(target)

def truenegative(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TN = torch.sum((1-pred) * (1-target)).item()
    return TN / len(target)

def falsepositive(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        FP = torch.sum(pred * (1-target)).item()
    return FP / len(target)

def falsenegative(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        FN = torch.sum((1-pred) * target).item()
    return FN / len(target)

def sensitivity(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TP = 0; FN = 0
        TP += torch.sum(pred * target).float().item()
        FN += torch.sum((1-pred) * target).item()
    return TP / (TP + FN + 1e-6)

def specificity(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TN = 0; FP = 0
        TN += torch.sum((1-pred) * (1-target)).float().item()
        FP += torch.sum(pred * (1-target)).float().item()
    return TN / (TN + FP + 1e-6)   

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        TP = 0; FP = 0
        TP += torch.sum(pred * target).float().item()
        FP += torch.sum(pred * (1 - target)).float().item()
    return TP / (TP + FP + 1e-6)

def F1(output, target, threshold=0.5):
    with torch.no_grad():
        # Sensitivity == Recall
        SE = sensitivity(output, target)
        PC = precision(output, target)
    return 2 * SE * PC / (SE + PC + 1e-6)