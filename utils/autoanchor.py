import random
from multiprocessing.pool import ThreadPool

import torch
import numpy as np
from tqdm import tqdm

from utils.general import TQDM_BAR_FORMAT, NUM_THREADS


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)

def gather_shapes_labels(dataset):
    shapes, labels = [None] * len(dataset), [None] * len(dataset)

    results = ThreadPool(NUM_THREADS).imap(dataset.get_GT, range(len(dataset)))
    for i, (_, lb, _, sh) in tqdm(enumerate(results), desc='Gathering bounding box data for Autoanchor...', 
                                  total=len(dataset), bar_format=TQDM_BAR_FORMAT):
        shapes[i], labels[i] = sh[:2], lb
    return np.array(shapes), labels

def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    shapes, labels = gather_shapes_labels(dataset=dataset)
    shapes = imgsz * shapes / shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(len(dataset), 1)) # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, labels)])).float() # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat
    
    m = model.head
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))

    s = f"\n{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:
        print(f'{s}Current anchors are a good fit to dataset')
    else:
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(shapes=shapes, labels=labels, n=na, 
                                img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)
            m.anchors /= stride
            s = 'Proceeding with new anchors'
        else:
            s = f'original anchors better than new anchors, proceeding with original anchors'
        print(s)


def kmean_anchors(shapes, labels, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        return x, x.max(1)[0]  # x, best_x
    
    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness
    
    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            print(s[:-2])
        return k

    # Get label wh
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, labels)])  # wh
    # Filter (object >= 2 pixels)
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)

    # Kmeans init
    try:
        print(f'Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        print(f'WARNING ! switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
            kg = (k.copy() * v).clip(min=2.0)
            fg = anchor_fitness(kg)
            if fg > f:
                f, k = fg, kg.copy()
                pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)