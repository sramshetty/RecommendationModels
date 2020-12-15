import torch

def get_recall(indices, targets, mask):
    item_recalls = {}

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
        
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()

    for item in targets:
        for hit in hits:
            x = targets[hit[0], hit[1]]
            if x == item:
                item_recalls[item] += 1

    recall = float(hits.size(0)) / float( mask.int().sum())

    return recall


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

    recall = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)

    return recall, mrr