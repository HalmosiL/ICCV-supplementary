import warnings
from distutils.version import LooseVersion
from typing import Callable, Dict, Optional, Union

import torch
from adv_lib.utils import BackwardCounter, ForwardCounter
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ConfusionMatrix
from forward import predict

def run_attack(
                model,
                loader,
                attack,
                image_list,
                target=None,
                metrics=_default_metrics,
                return_adv=False
            ):
    device = next(model.parameters()).device
    targeted = False
    loader_length = len(loader)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, apsrs, apsrs_orig = [], [], [], []
    distances = {k: [] for k in metrics.keys()}

    if return_adv:
        images, adv_images = [], []

#####################################################################################################################
    for i, (images, labels) in enumerate(tqdm(loader, ncols=80, total=loader_length)):
        logits_arr = []
        labels_arr = []
        attack_label_arr = []

##############################################-NORMAL-TEST-BLOCK###################################################
        for k in range(len(images)):
            image = images[k]
            label = labels[k]

            if return_adv:
                images.append(image.clone())

            image, label = image.to(device), label.to(device).squeeze(1).long()

            if targeted:
                if isinstance(target, Tensor):
                    attack_label_arr.append(target.to(device).expand(image.shape[0], -1, -1))
                elif isinstance(target, int):
                    attack_label_arr.append(torch.full_like(label, fill_value=target))
            else:
                attack_label_arr.append(label)

            log_pred = predict(
              model=model,
              image=image,
              target=label,
              device=device,
              attack=None
            )
                
            logits_arr.append(log_pred)
            labels_arr.append(label)

        logits = torch.zeros(19, 898, 1796)
        label = torch.zeros(1, 898, 1796)
        attack_label = torch.zeros(1, 898, 1796)

        d = 0

        for x in range(2):
            for y in range(4):
                logits[:, x*449:(x+1)*449, y*449:(y+1)*449] = logits_arr[d]
                label[:, x*449:(x+1)*449, y*449:(y+1)*449] = labels_arr[d]
                attack_label[:, x*449:(x+1)*449, y*449:(y+1)*449] = attack_label_arr[d]
                d += 1

        logits = logits.reshape(1, 19, 898, 1796).to(device)

        attack_label = attack_label.to(device)
        label = label.to(device)

        if i == 0:
            num_classes = logits.size(1)
            confmat_orig = ConfusionMatrix(num_classes=num_classes)
            confmat_adv = ConfusionMatrix(num_classes=num_classes)

        mask = label < num_classes
        mask_sum = mask.flatten(1).sum(dim=1)
        pred = logits.argmax(dim=1)

        accuracies.extend(((pred == label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())
        confmat_orig.update(label, pred)

        if targeted:
            target_mask = attack_label < logits.size(1)
            target_sum = target_mask.flatten(1).sum(dim=1)
            apsrs_orig.extend(((pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs_orig.extend(((pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())

        forward_counter.reset(), backward_counter.reset()
        acc_global, accs, ious = confmat_orig.compute()
#####################################################################################################################

        start.record()

        adv_images_arr = []

        for k in range(len(images)):
            image = images[k]
            label_ = labels[k]

            image, label_ = image.to(device), label_.to(device).squeeze(1).long()
            
            _, adv_image = predict(
              model=model,
              image=image,
              target=attack_label_arr[k],
              device=device,
              attack=attack
            )
            
            adv_images_arr.append(adv_image)

        # performance monitoring
        end.record()
        torch.cuda.synchronize()
        times.append((start.elapsed_time(end)) / 1000)  # times for cuda Events are in milliseconds
        forwards.append(forward_counter.num_samples_called)
        backwards.append(backward_counter.num_samples_called)
        forward_counter.reset(), backward_counter.reset()

        adv_logits_arr = []

        for k in range(len(adv_images_arr)):
            if adv_images_arr[k].min() < 0 or adv_images_arr[k].max() > 1:
                warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
                adv_images_arr[k].clamp_(min=0, max=1)

            if return_adv:
                adv_images.append(adv_images_arr[k].cpu().clone())
                
            log_pred = predict(
              model=model,
              image=adv_images_arr[k],
              target=attack_label_arr[k],
              device=device,
              attack=None
            )
                
            adv_logits_arr.append(log_pred)

        adv_pred = torch.zeros(19, 898, 1796).to(device)
        image_full = torch.zeros(3, 898, 1796).to(device)
        adv_image_full = torch.zeros(3, 898, 1796).to(device)

        d = 0

        for x in range(2):
            for y in range(4):
                adv_pred[:, x*449:(x+1)*449, y*449:(y+1)*449] = adv_logits_arr[d]
                image_full[:, x*449:(x+1)*449, y*449:(y+1)*449] = images[k]
                adv_image_full[:, x*449:(x+1)*449, y*449:(y+1)*449] = adv_images_arr[k]

                d += 1

        adv_pred = adv_pred.reshape(1, 19, 898, 1796)
        adv_pred = adv_pred.argmax(dim=1)

        confmat_adv.update(label, adv_pred)

        if targeted:
            apsrs.extend(((adv_pred == attack_label) & target_mask).flatten(1).sum(dim=1).div(target_sum).cpu().tolist())
        else:
            apsrs.extend(((adv_pred != label) & mask).flatten(1).sum(dim=1).div(mask_sum).cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_image_full, image_full).detach().cpu().tolist())

        acc_global, accs, ious = confmat_orig.compute()
        adv_acc_global, adv_accs, adv_ious = confmat_adv.compute()

######################################################################################################################

    data = {
        'image_names': image_list[:len(apsrs)],
        'targeted': targeted,
        'accuracy': accuracies,
        'acc_global': acc_global.item(),
        'adv_acc_global': adv_acc_global.item(),
        'ious': ious.cpu().tolist(),
        'adv_ious': adv_ious.cpu().tolist(),
        'apsr_orig': apsrs_orig,
        'apsr': apsrs,
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': distances,
    }

    if return_adv:
        shapes = [img.shape for img in images]
        if len(set(shapes)) == 1:
            images = torch.cat(images, dim=0)
            adv_images = torch.cat(adv_images, dim=0)
        data['images'] = images
        data['adv_images'] = adv_images

    return data
