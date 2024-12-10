from models.dave import build_model
from models.box_prediction import BoxList, boxlist_nms
from utils.data import FSC147WithDensityMapDOWNSIZE
from utils.arg_parser import get_argparser
from utils.losses import Criterion, Detection_criterion
from time import perf_counter
import argparse
import os, json
from torchvision.ops import box_iou
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import numpy as np
import skimage
import math
from tqdm import tqdm

DATASETS = {
    "fsc147": FSC147WithDensityMapDOWNSIZE,
}


def generate_bbox(density_map, tlrb):
    bboxes = []
    for i in range(density_map.shape[0]):
        density = np.array((density_map)[i][0].cpu())
        dmap = np.array((density_map)[i][0].cpu())

        mask = dmap < np.max(dmap) / 3
        dmap[mask] = 0
        a = skimage.feature.peak_local_max(dmap, exclude_border=0)

        boxes, scores= [], []
        b, l, r, t = tlrb[i]

        for x11, y11 in a:
            box = [ y11 - b[x11][y11].item(),
                    x11 - l[x11][y11].item(),
                    y11 + r[x11][y11].item(),
                    x11 + t[x11][y11].item()]
            boxes.append(box)
            scores.append(1 - math.fabs(density[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])].sum() - 1))

        b = BoxList(boxes, (density_map.shape[3], density_map.shape[2]))
        b.fields["scores"] = torch.tensor(scores)
        b = b.clip()
        b = boxlist_nms(b, b.fields["scores"], 0.55)

        bboxes.append(b)
    return bboxes


def reduce_dict(input_dict, average=False):
    with torch.no_grad():
        names, values = [], []
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert args.backbone in ["resnet18", "resnet50", "resnet101"]
    assert args.reduction in [4, 8, 16]

    model = build_model(args).to(device)

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + ".pth"))["model"],
        strict=False,
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

    # Freeze all parameter
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Only train the box_predictor
    for name, param in model.named_parameters():
        if "box_predictor" in name:
            param.requires_grad = True

    optimizer = torch.optim.AdamW([{"params": fcos_params.values(), "lr": args.lr},],
                                    lr=args.lr, weight_decay=args.weight_decay,)
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
        best = 10000000000000
        best_mAP = 0

    criterion = Criterion(args)
    aux_criterion = Criterion(args, aux=True)
    det_criterion = Detection_criterion([[-1, args.fcos_pred_size],
                                        [64, 128],
                                        [128, 256],
                                        [256, 512],
                                        [512, 100000000],], # config.sizes,
                                        "giou",             # config.iou_loss_type,
                                        True,               # config.center_sample,
                                        [1],                # config.fpn_strides,
                                        5,)                  # config.pos_radius,
    # Load Dataset
    train = DATASETS[args.dataset]( args.data_path,
                                    args.image_size,
                                    split="train",
                                    num_objects=args.num_objects,
                                    tiling_p=args.tiling_p,
                                    zero_shot=args.zero_shot or args.orig_dmaps,
                                    skip_cars=args.skip_cars,)
    val = DATASETS[args.dataset](   args.data_path,
                                    args.image_size,
                                    split="val",
                                    num_objects=args.num_objects,
                                    tiling_p=args.tiling_p,
                                    zero_shot=args.zero_shot or args.orig_dmaps,)
    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers,)
    val_loader = DataLoader( val, shuffle=False, batch_size=args.batch_size, drop_last=False, num_workers=args.num_workers,)
    # Change Path to save the log
    det_logdir = "./det_logdir_3"
    os.makedirs(det_logdir, exist_ok=True)
    print("NUM STEPS", len(train_loader) * args.epochs)
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        start               = perf_counter()
        train_losses        = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        val_losses          = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        aux_train_losses    = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        aux_val_losses      = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        train_ae            = torch.tensor(0.0).to(device)
        val_ae              = torch.tensor(0.0).to(device)
        mAP                 = torch.tensor(0.0).to(device)

        model.train()
        with tqdm(train_loader, desc="Training Progress", unit="batch") as pbar:
            for img, bboxes, density_map, ids, scale_x, scale_y, _ in pbar:
                img     = img.to(device)
                bboxes  = bboxes.to(device)
                density_map = density_map.to(device)
                targets = ( BoxList(bboxes, (args.image_size, args.image_size), mode="xyxy").to(device)
                            .resize((args.fcos_pred_size, args.fcos_pred_size)))
                targets.fields["labels"] = [1 for __ in range(args.batch_size * 2)]
                optimizer.zero_grad()

                outR, aux_R, tblr, location = model(img, bboxes)

                if args.normalized_l2:
                    with torch.no_grad():
                        num_objects = density_map.sum()
                else:
                    num_objects = None

                main_losses = criterion(outR, density_map, bboxes, num_objects)
                aux_losses  = [aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_R]
                det_loss    = det_criterion(location, tblr, targets)
                del targets
                loss = (sum([ml for ml in main_losses.values()]) * 0
                        + sum([al for alls in aux_losses for al in alls.values()]) * 0
                        + det_loss)  # + l

                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                train_losses        = { k: train_losses[k] + main_losses[k] * img.size(0) for k in train_losses.keys()}
                aux_train_losses    = { k: aux_train_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                                        for k in aux_train_losses.keys()}

                train_ae += torch.abs(density_map.flatten(1).sum(dim=1) - outR.flatten(1).sum(dim=1)).sum()
                # tqdm display the variation of loss
                pbar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation Progress", unit="batch") as pbar:
                for img, bboxes, density_map, ids, scale_x, scale_y, _ in pbar:
                    gt_bboxes, _    = val.get_gt_bboxes(ids)
                    img             = img.to(device)
                    bboxes          = bboxes.to(device)
                    density_map     = density_map.to(device)
                    optimizer.zero_grad()
                    outR, aux_R, tblr, location = model(img, bboxes)
                    boxes_pred                  = generate_bbox(outR, tblr)

                    for iii in range(len(gt_bboxes)):
                        boxes_pred[iii].box = (boxes_pred[iii].box * 1
                            / torch.tensor([scale_y[iii], scale_x[iii], scale_y[iii], scale_x[iii]]))
                        mAP += (box_iou(gt_bboxes[iii], boxes_pred[iii].box).max(dim=1)[0].sum()
                            / gt_bboxes[iii].shape[1])

                    if args.normalized_l2:
                        with torch.no_grad():
                            num_objects = density_map.sum()
                    else:
                        num_objects = None
                    main_losses = criterion(outR, density_map, bboxes, num_objects)
                    aux_losses  = [aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_R]
                    val_losses  = {k: val_losses[k] + main_losses[k] * img.size(0) for k in val_losses.keys()}
                    aux_val_losses = {  k: aux_val_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                                        for k in aux_val_losses.keys()}
                    val_ae += torch.abs(density_map.flatten(1).sum(dim=1) - outR.flatten(1).sum(dim=1)).sum()

                    # tqdm display the variation of loss
                    pbar.set_postfix(loss=loss.item())

        scheduler.step()
        end = perf_counter()
        best_epoch = False
        # Change Path to save the checkpoint
        path_dir = os.path.join(args.model_path, "detection_3")
        os.makedirs(path_dir, exist_ok=True)
        path_name = os.path.join(path_dir, f'{args.det_model_name}_{epoch}.pth')
        trainable_state_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
        checkpoint = {  "epoch": epoch,
                        "box_predictor": trainable_state_dict,  # Save only the trainable parameters
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_ae": val_ae.item() / len(val),}
        torch.save(checkpoint, path_name)
        print("Epoch", epoch)
        print("Train loss", {k: v.item() / len(train) for k, v in train_losses.items()})
        print("Val loss", {k: v.item() / len(val) for k, v in val_losses.items()})
        print("Train Aux loss", {k: v.item() / len(train) for k, v in aux_train_losses.items()})
        print("Test Aux loss", {k: v.item() / len(val) for k, v in aux_val_losses.items()})

        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = True

        print(  train_ae.item() / len(train), val_ae.item() / len(val), "best" if best_epoch else "")
        print("det_sc:", mAP / len(val))
        print("*****************************************")

        epoch_data = {  "epoch": epoch,
                        "Train loss": {k: v.item() / len(train) for k, v in train_losses.items()},
                        "Val loss": {k: v.item() / len(val) for k, v in val_losses.items()},
                        "Train Aux loss": {k: v.item() / len(train) for k, v in aux_train_losses.items()},
                        "Test Aux loss": {k: v.item() / len(val) for k, v in aux_val_losses.items()},
                        "Train ae": train_ae.item() / len(train),
                        "Val ae": val_ae.item() / len(val),
                        "best_epoch": best_epoch,
                        "best_model_path": path_name if (best_epoch == True) else None }

        json_filename = os.path.join(det_logdir, f"det_{epoch}.json")
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
