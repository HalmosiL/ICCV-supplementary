from executor.Adversarial import model_immer_attack_auto_loss_combination
from torchvision import transforms
import torch
import logging
import time

def run(id_, batch, device, model, attack, number_of_steps, data_queue, split, split_size, epoch, gen=True, methods="Normal"):
    logging.debug("Gen_" + str(id_) + " started..")
    
    image = batch[0].to(device)
    label = batch[1].to(device)

    print(image.shape)
    print(label.shape)

    image_adversarial = model_immer_attack_auto_loss_combination(
        image=image,
        target=label,
        model=model,
        attack=attack,
        number_of_steps=number_of_steps,
        device=device
    )

    torch.save(image_adversarial.cpu().detach(), data_queue + 'image_' + str(id_) + '_0_.pt')
    torch.save(label.cpu().detach(), data_queue + 'label_' + str(id_) + '_0_.pt')
