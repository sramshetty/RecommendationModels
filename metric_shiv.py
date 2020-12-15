import torch
from collections import defaultdict

def get_recall(indices, targets, mask):
    item_recalls = defaultdict(float)
    item_counts = defaultdict(int)

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
        
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()

    items = targets[:,0]
    for i in range(len(items)):
        if mask[i]:
            item_counts[items[i].item()] += 1
    
    for x in hits[:,0]:
        item = targets[x,0]
        item_recalls[item.item()] += 1

    for item,hit in item_recalls.items():
        item_recalls[item] = float(hit) / float(item_counts[item])

    recall = float(hits.size(0)) / float( mask.int().sum())

    return recall, item_recalls


def get_mrr(indices, targets, mask):
    item_mrrs = defaultdict(float)
    item_counts = defaultdict(int)

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()

    items = targets[:,0]
    for i in range(len(items)):
        if mask[i]:
            item_counts[items[i].item()] += 1

    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    
    item_hits = hits[:,0]
    for x in range(len(item_hits)):
        item = targets[item_hits[x],0]
        item_mrrs[item.item()] += rranks[x].item()

    for item,rank in item_mrrs.items():
        item_mrrs[item] = float(rank) / float(item_counts[item])

    mrr = torch.sum(rranks).data / float( mask.int().sum())
    return mrr.item(), item_mrrs


def evaluate(indices, targets, mask, k=20, debug=False):
    _, indices = torch.topk(indices, k, -1)
    
    indices = indices.cpu()
    targets = targets.cpu()

    recall, item_recalls = get_recall(indices, targets, mask)
    mrr, item_mrrs = get_mrr(indices, targets, mask)

    return recall, item_recalls, mrr, item_mrrs