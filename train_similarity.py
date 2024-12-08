import argparse
import os, json
from time import perf_counter

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from models.dave_tr import build_model
from utils.arg_parser import get_argparser
from utils.data import FSC147WithDensityMapSimilarityStitched
from utils.losses import Criterion
from tqdm import tqdm
DATASETS = {
    "fsc147": FSC147WithDensityMapSimilarityStitched,
}


def reduce_dict(input_dict, average=False):
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train(args):

    if args.skip_train:
        print("SKIPPING TRAIN")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert args.backbone in ["resnet18", "resnet50", "resnet101"]
    assert args.reduction in [4, 8, 16]

    model = build_model(args).to(device)

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + '.pth'))['model'], strict=False
    )

    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    feat_comp = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in n:
            backbone_params[n] = p
        elif "box_predictor" in n:
            fcos_params[n] = p
        elif "feat_comp" in n:
            feat_comp[n] = p
        else:
            non_backbone_params[n] = p

    pretrained_dict_feat = {k.split("backbone.backbone.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, args.model_name+'.pth'))[
                                'model'].items() if 'backbone' in k}
    model.backbone.backbone.load_state_dict(pretrained_dict_feat)    

    optimizer = torch.optim.AdamW(
        [{"params": feat_comp.values()}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f"{args.model_name}.pth"))
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        best = checkpoint["best_val_ae"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        start_epoch = 0
        best = 10000000

    criterion = Criterion(args)
    aux_criterion = Criterion(args, aux=True)

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="train",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
    )

    train_loader = DataLoader(
        train,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split="val",
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
    )
    val_loader = DataLoader(
        val,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )
    # Change Path to save the log
    sim_logdir = "./sim_logdir_2"
    os.makedirs(sim_logdir, exist_ok=True)
    print("NUM STEPS", len(train_loader) * args.epochs)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)

        model.train()

        for img, bboxes, indices, density_map,  img_ids in tqdm(train_loader, desc="Training Progress", unit="batch"):
            img = img.to(device)
            bboxes = bboxes.to(device)
            optimizer.zero_grad()
            loss, _ = model(img, bboxes)

            train_loss += loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        print("VALIDATION")
        model.eval()
        with torch.no_grad():
            data_num = 0
            for img, bboxes,indices, density_map, img_ids in tqdm(val_loader, desc="Validation Progress", unit="batch"):
                img = img.to(device)
                bboxes = bboxes.to(device)

                optimizer.zero_grad()
                loss, _ = model(img, bboxes)
                val_loss += loss
                data_num += 1

        scheduler.step()
        end = perf_counter()
        best_epoch = False
        checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_ae': val_loss.item() / len(val)
        }
        # Change Path to save the checkpoint
        path_dir = os.path.join(args.model_path, "similarity_2")
        os.makedirs(path_dir, exist_ok=True)
        path_name = os.path.join(path_dir, f'{args.det_model_name}_{epoch}.pth')
        torch.save( checkpoint, path_name)
        print("Epoch", epoch)
        print("Train loss", train_loss.item() / len(train))
        print("Val loss", val_loss.item() / len(val))
        if val_loss.item()  < best:
            best = val_loss
            best_epoch = True
            if best_epoch :
                print('Best Epoch : ', epoch)
                print('Best Model is saved to :',path_name)
        print("*****************************************")

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss.item() / len(train),
            "val_loss": val_loss.item() / len(val),
            "best_epoch": best_epoch,
            "best_model_path": path_name if (best_epoch == True) else None
        }
        json_filename = os.path.join(sim_logdir, f"sim_{epoch}.json")
        with open(json_filename, 'w') as json_file:
            json.dump(epoch_data, json_file, indent=4)
        print(f"Epoch {epoch} information saved to {json_filename}")

    if args.skip_test:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DAVE", parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)
    # "train_length": 3659,
    # "val_length": 1286,
