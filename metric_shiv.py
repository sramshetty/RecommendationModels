import torch
from collections import defaultdict

def get_recall(indices, targets, mask):
    item_recalls = defaultdict(int)
    item_counts = defaultdict(int)

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
        
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()

    for item in targets[:,0]:
        item_counts[item] += 1
        if item_recalls[item] == 0:
            for i, j in hits:
                print(i, j)
                x = targets[i, j]
                print(x, item)
                if x == item:
                    item_recalls[item] += 1

    recall = float(hits.size(0)) / float( mask.int().sum())

    return recall, item_recalls, item_counts


def get_mrr(indices, targets, mask):
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices)
    hits *= mask.view(-1, 1).expand_as(indices)
    
    hits = hits.nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / float( mask.int().sum())
    return mrr.item()


def evaluate(indices, targets, mask, k=20, debug=False):
    _, indices = torch.topk(indices, k, -1)
    
    indices = indices.cpu()
    targets = targets.cpu()

    recall, item_recall, item_counts = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)

    return recall, item_recall, item_counts, mrr