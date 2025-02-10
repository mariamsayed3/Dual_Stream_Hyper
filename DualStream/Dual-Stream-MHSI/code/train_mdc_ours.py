import os
import torch
from torch.utils.data import DataLoader
import json
from Data_Generate import Data_Generate_Transc
from basemodel import SST_Seg_Dual
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
from segmentation_models_pytorch.utils.losses import DiceLoss
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import confusion_matrix


def train_one_epoch(model, train_loader, optim, criterion, device):
    model.train()
    train_loss = 0
    for i, (data, target) in enumerate(tqdm(train_loader)):
        # print(data.shape, target.shape)
        data, target = data.to(device), target.long().to(device)
        optim.zero_grad()
        print('model')
        output = model(data)
        print('criterion')
        loss = DiceLoss()(output, F.one_hot(target, num_classes=5).permute(0, 3, 1, 2))
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss


class Dice_CE_Loss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(Dice_CE_Loss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        dice_loss_value = self.dice_loss(input, target)
        ce_loss_value = self.ce_loss(input, target)
        combined_loss = (
            self.ce_weight * ce_loss_value + self.dice_weight * dice_loss_value
        )
        return combined_loss


root_path = "./bioDataset2"
dataset = "./dataset/train_val_test_transc.json"
with open(dataset, "r") as load_f:
    dataset_dict = json.load(load_f)

# print(dataset_dict)
train_files = dataset_dict["train"]
val_files = dataset_dict["val"]

train_gene_path = [f"{root_path}/gene_expre_matrix_{i}.npy" for i in train_files]
train_labels_path = [f"{root_path}/cell_type_matrix{i}.npy" for i in train_files]
val_gene_path = [f"{root_path}/gene_expre_matrix_{i}.npy" for i in val_files]
val_labels_path = [f"{root_path}/cell_type_matrix{i}.npy" for i in val_files]

# print(
#     len(train_gene_path),
#     len(train_labels_path),
#     len(val_gene_path),
#     len(val_labels_path),
# )

train_dataset = Data_Generate_Transc(train_gene_path, train_labels_path)
train_loader = DataLoader(
    train_dataset, batch_size=3
)  # bigger batch size when on kaggle
val_dataset = Data_Generate_Transc(val_gene_path, val_labels_path)
val_loader = DataLoader(val_dataset, batch_size=3)

# print(len(train_dataset), len(val_dataset))


# ex = next(iter(train_loader))
# print(len(ex))
# print(ex[0].shape, ex[1].shape)


device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# torch.cuda.set_device(device)
# device = torch.device('cuda', local_rank)
# torch.distributed.init_process_group(backend='nccl')

# spectral_channels = 9718  # may be reduced
# out_channels = 5

model = SST_Seg_Dual(
    spectral_channels=288,
    out_channels=5,
    spectral_hidden_feature=64,
    spatial_pretrain=False,
    decode_choice="unet",
    backbone="resnet34",
    bands_group=144,
    linkpos=[0, 0, 1, 0, 1, 0],
    spe_kernel_size=1,
    spa_reduction=[4, 4],
    merge_spe_downsample=[2, 2],
    hw=[32, 32],
    rank=4,
    attention_group="non",
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

#assert False

# print(model(ex[0].to(device)).argmax(1).shape)


# Problem 2: Links and band groups --> today at hackathon

EPOCHS = 0
# Optimizer, Scheuler and Loss

lr = 3e-4
wd = 5e-4
optim = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd
)

scheduler = CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-8)

criterion = Dice_CE_Loss(ce_weight=0.5, dice_weight=0.5)

# Training Loop
history = {
    "epoch": [],
    "lr": [],
    "train_loss": [],
    "val_loss": [],
}

# Training & Validation loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}/{EPOCHS}")
    try:
        train_loss = train_one_epoch(model, train_loader, optim, criterion, device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("| WARNING: ran out of memory")
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            break
        else:
            raise e

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader)):
            data, target = data.to(device), target.long().to(device)
            output = model(data)
            loss = DiceLoss()(output, F.one_hot(target, num_classes=5).permute(0, 3, 1, 2))
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss}")
    scheduler.step()
    history["epoch"].append(epoch)
    history["lr"].append(scheduler.get_last_lr())
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    torch.save(model.state_dict(), f"./model_{epoch}.pth")
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("Model & History Saved")


model.load_state_dict(torch.load('trained_on_model_9.pth'))
model.eval()
val_loss = 0
targets_arr = []
outputs_arr = []
with torch.no_grad():
    for i, (data, target) in enumerate(tqdm(val_loader)):
        data, target = data.to(device), target.long().to(device)
        output = model(data)
        targets_arr.append(target)
        outputs_arr.append(output)

targets = torch.cat(targets_arr, dim=0)
outputs = torch.cat(outputs_arr, dim=0)
print(targets.shape, outputs.shape)
print(targets.cpu().flatten().numpy().shape, outputs.argmax(1).cpu().flatten().numpy().shape)
cm = confusion_matrix(targets.cpu().flatten().numpy(), outputs.argmax(1).cpu().flatten().numpy())
cm_df = pd.DataFrame(cm)

# Save the DataFrame to a CSV file
cm_df.to_csv("confusion_matrix2.csv", index=False)


