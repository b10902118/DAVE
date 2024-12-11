import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from models.dave import build_model
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.arg_parser import get_argparser
from utils.data import (
    FSC147WithDensityMapSCALE2BOX,
    pad_image,
    FSC147WithDensityMapDOWNSIZE,
)
from utils.data_lvis import FSCD_LVIS_Dataset_SCALE

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from models.box_prediction import BoxList
import multiprocessing
import time
from torchvision.ops import box_iou

# import psutil # debug open file

DATASETS = {
    "fsc_box": FSC147WithDensityMapSCALE2BOX,
    "fsc_downscale": FSC147WithDensityMapDOWNSIZE,
    "lvis": FSCD_LVIS_Dataset_SCALE,
}


def to_cpu_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy()
    else:
        return arr


def draw_bbox(boxes, color, width=1):
    for box in boxes:
        plt.plot(
            [box[0], box[0], box[2], box[2], box[0]],
            [box[1], box[3], box[3], box[1], box[1]],
            linewidth=width,
            color=color,
        )


def visualize(
    idx,
    image_path,
    dmap,
    dmap_no_mask,
    pred_boxes_resized,
    pred_boxes_no_resized,
    gt_boxes,
    save_dir,
    title_counts,
):

    if not torch.equal(dmap, dmap_no_mask):
        plt.figure(figsize=(30 / 2, 10 / 2))
        n_cols = 3
    else:
        plt.figure(figsize=(20 / 2, 10 / 2))
        n_cols = 2

    # Subplot for bounding boxes
    plt.subplot(1, n_cols, 1)
    image = plt.imread(image_path)
    plt.imshow(image)
    draw_bbox(pred_boxes_resized, "red")
    draw_bbox(gt_boxes, "blue", width=2)

    # Subplot for density map
    plt.subplot(1, n_cols, 2)
    dmap_np = to_cpu_numpy(dmap)
    plt.imshow(dmap_np, cmap="viridis")
    plt.colorbar()
    draw_bbox(pred_boxes_no_resized, "red", width=0.5)

    if n_cols == 3:
        plt.subplot(1, n_cols, 3)
        dmap_no_mask_np = to_cpu_numpy(dmap_no_mask)
        plt.imshow(dmap_no_mask_np, cmap="viridis")
        draw_bbox(pred_boxes_no_resized, "red", width=0.5)

    plt.suptitle(
        f"{os.path.basename(image_path)} {image.shape[:2]}\n{title_counts}", fontsize=20
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{idx}.png")
    plt.close()


def plotting_process(plot_queue):
    while True:
        args = plot_queue.get()
        if args is None:
            break
        # print("Plotting", args[0], flush=True)
        visualize(*args)
        # print("Done plotting", args[0], flush=True)
        plot_queue.task_done()


def generate_annotations(boxes_pred, image_id):
    if not hasattr(generate_annotations, "anno_id"):
        generate_annotations.anno_id = 1  # Initialize the static anno_id
    areas = boxes_pred[0].area()
    boxes_xywh = boxes_pred[0].convert("xywh")
    img_info = {
        "id": image_id
        # "file_name": "None",
    }
    scores = boxes_xywh.fields["scores"]
    annos = []
    for i in range(len(boxes_pred[0].box)):
        box = boxes_xywh.box[i]
        anno = {
            "id": generate_annotations.anno_id,
            "image_id": image_id,
            "area": int(areas[0].item()),
            "bbox": [
                int(box[0].item()),
                int(box[1].item()),
                int(box[2].item()),
                int(box[3].item()),
            ],
            "category_id": 1,
            "score": float(scores[i].item()),
        }
        generate_annotations.anno_id += 1
        annos.append(anno)
    return annos, img_info


@torch.no_grad()
def evaluate(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + ".pth"))["model"],
        strict=False,
    )
    pretrained_dict_feat = {
        k.split("feat_comp.")[1]: v
        for k, v in torch.load(os.path.join(args.model_path, "verification.pth"))[
            "model"
        ].items()
        if "feat_comp" in k
    }
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)

    # just one, multiple processes stuck
    plot_queue = multiprocessing.JoinableQueue()
    plot_process = multiprocessing.Process(target=plotting_process, args=(plot_queue,))
    plot_process.start()

    bigs = {"val": [229, 437, 438, 440, 447, 456, 1181], "test": [120, 1057]}
    masked = {"val": [], "test": []}

    # custom_indices = {"val": [i for i in range(117, 128)], "test": []}
    custom_indices = {
        "val": None,  # [],  # [i for i in range(1286) if i not in bigs["val"]],
        "test": None,  # [i for i in range(1190) if i not in bigs["test"]],
    }

    normalized = {"val": [], "test": []}

    for split in ["val", "test"]:
        print("Evaluating", split)
        generate_annotations.anno_id = 1
        save_dir = f"output_images/{split}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        test: FSC147WithDensityMapSCALE2BOX = DATASETS["fsc_box"](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
            custom_indices=custom_indices[split],
        )

        test_loader = DataLoader(
            test,
            batch_size=1,
            drop_last=False,
            num_workers=args.num_workers,
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)

        model.eval()

        predictions = {"images": [], "annotations": []}
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #    futures = []
        semi_incontext = False
        draw = False
        semi_diff = {}
        for (
            img,
            ex_bboxes,  # examples only
            _,
            ids,  # <class 'torch.Tensor'> tensor([i]) # caused by batch_size=1
            scale_x,
            scale_y,
            shape,
        ) in tqdm(test_loader):
            # inference
            idx = ids[0].item()
            img = img.to(device)
            # ex_bboxes = torch.cat([ex_bboxes, ex_bboxes], dim=1)
            ex_bboxes = ex_bboxes.to(device)
            # for box in ex_bboxes[0]:
            #    x_mid = (box[0] + box[2]) / 2
            #    y_mid = (box[1] + box[3]) / 2
            #    box[0] = (box[0] + x_mid) / 2
            #    box[1] = (box[1] + y_mid) / 2
            #    box[2] = (box[2] + x_mid) / 2
            #    box[3] = (box[3] + y_mid) / 2

            # record big images
            # start_time = time.time()  # Start time measurement
            out, out_no_mask, tblr, pred_boxes = model(
                img, ex_bboxes, test.image_names[idx]
            )
            if semi_incontext:
                orig_density_map = out[0, 0, : shape[1], : shape[2]].cpu()
                orig_pred_dcnt = orig_density_map.sum().item()

                candidate_boxes = pred_boxes.box[pred_boxes.fields["scores"] > 0.95]

                candidate_boxes = candidate_boxes.to(device)
                iou_threshold = 0.5
                keep_boxes = []
                for pred_box in candidate_boxes:
                    iou = box_iou(pred_box.unsqueeze(0), ex_bboxes[0]).max().item()
                    if iou <= iou_threshold:
                        keep_boxes.append(pred_box)

                # keep_density_boxes = []
                # # filter by density > 0.9
                # for bbox in keep_boxes:
                #     x_min, y_min, x_max, y_max = map(int, bbox)
                #     cropped_density = orig_density_map[y_min:y_max, x_min:x_max]
                #     object_count = cropped_density.sum().item()
                #     if object_count > 0.9:
                #         keep_density_boxes.append(bbox)

                keep_density_boxes = []
                density_diffs = []

                # Filter by density > 0.9 and calculate the absolute difference from 1
                for bbox in keep_boxes:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    cropped_density = orig_density_map[y_min:y_max, x_min:x_max]
                    object_count = cropped_density.sum().item()
                    if object_count > 0.9:
                        keep_density_boxes.append(bbox)
                        # density_diffs.append(abs(1 - object_count))

                # Sort keep_density_boxes by the absolute difference from 1 in density
                # keep_density_boxes = [
                #    box for _, box in sorted(zip(density_diffs, keep_density_boxes))
                # ]

                # filter pred boxes by iou # not observed problem
                # filtered_boxes = []
                # for i, box in enumerate(keep_boxes):
                #     keep = True
                #     for j in range(i):
                #         if box_iou(box.unsqueeze(0), keep_boxes[j].unsqueeze(0)).item() > 0.5:
                #             keep = False
                #             break
                #     if keep:
                #         filtered_boxes.append(box)
                # keep_boxes = filtered_boxes

                if keep_density_boxes:
                    candidate_boxes = torch.stack(keep_density_boxes)
                    n = orig_pred_dcnt
                    ex_max = 1  # min(round(n / 5), 3)
                    candidate_boxes = candidate_boxes[:ex_max]
                    candidate_boxes = torch.unsqueeze(candidate_boxes, 0)
                    candidate_boxes = candidate_boxes.to(device)
                    ex_bboxes = torch.cat([ex_bboxes, candidate_boxes], dim=1)
                    out, out_no_mask, tblr, pred_boxes = model(
                        img, ex_bboxes, test.image_names[idx]
                    )
            # elapsed_time = time.time() - start_time  # End time measurement

            # if elapsed_time > 60:
            #    bigs[split].append((idx, test.image_names[idx]))

            # record masked images
            # if not torch.equal(out, out_no_mask):
            #    masked[split].append((idx, test.image_names[idx]))

            pred_boxes: BoxList = pred_boxes.to("cpu")

            # resize pred boxes
            _, resize_factors = test.get_gt_bboxes(ids)
            scale = torch.tensor([scale_y[0], scale_x[0], scale_y[0], scale_x[0]]).cpu()
            pred_boxes_no_resized = pred_boxes.box.clone()  # for heatmap
            pred_boxes.box = pred_boxes.box / scale * resize_factors[0]
            pred_boxes_resized = pred_boxes.box  # for image

            # resize gt boxes
            gt_boxes = ex_bboxes[0].cpu() / scale * resize_factors[0].cpu()

            # generate annotations
            image_id = test.map_img_name_to_ori_id()[test.image_names[idx]]
            annos, img_info = generate_annotations([pred_boxes], image_id)
            predictions["annotations"].append(annos)
            predictions["images"].append(img_info)

            # compute error
            density_map = out[0, 0, : shape[1], : shape[2]].cpu()
            density_map_no_mask = out_no_mask[0, 0, : shape[1], : shape[2]].cpu()
            gt_cnt = test.get_gt_count(idx)

            total_ex_density_sum = 0
            for bbox in ex_bboxes[0]:
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped_density = density_map[y_min:y_max, x_min:x_max]
                total_ex_density_sum += cropped_density.sum().item()

            ex_avg_density_sum = total_ex_density_sum / 3
            if ex_avg_density_sum > 1.5:  # normalizing the density map
                normalized[split].append((idx, ex_avg_density_sum))
                density_map /= ex_avg_density_sum
                density_map_no_mask /= ex_avg_density_sum

            pred_dcnt = density_map.sum().item()
            pred_dcnt_no_mask = density_map_no_mask.sum().item()
            pred_bcnt = len(pred_boxes.box)
            ae += abs(gt_cnt - pred_dcnt)
            se += (gt_cnt - pred_dcnt) ** 2

            if semi_incontext:
                diff = abs(gt_cnt - orig_pred_dcnt) - abs(gt_cnt - pred_dcnt)
                semi_diff[idx] = round(diff, 2)
                win_str = f"Win: {semi_diff[idx]}"
                # print(win_str)

            # visualization
            if idx == 126:
                image_path = test.get_image_path(idx)
                title_counts = f"Dmap: {pred_dcnt:.1f} Box: {pred_bcnt} GT: {gt_cnt} NormalizingFactor: {ex_avg_density_sum:.1f}"
                arg = (
                    idx,
                    image_path,
                    density_map,
                    density_map_no_mask,
                    pred_boxes_resized,
                    pred_boxes_no_resized,
                    gt_boxes,
                    save_dir,
                    title_counts,
                )
                plot_queue.put(arg)

            # print(f"Added {idx} to queue", flush=True)
            # futures.append(executor.submit(visualize_wrapper, args))
            # visualize(
            #    idx,
            #    image,
            #    density_map,
            #    density_map_no_mask,
            #    pred_boxes_resized,
            #    pred_boxes_no_resized,
            #    gt_boxes,
            #    save_dir,
            #    title,
            # )
            # visualize_bboxes(idx, image, pred_boxes, gt_boxes, bbox_dir, title)
            # visualize_density_map(idx, density_map, density_dir, title=title)

        print(
            f"#{split} MAE {ae.item() / len(test)} RMSE {torch.sqrt(se / len(test)).item()}"
        )

        if semi_incontext:
            with open("./semi_" + split + ".json", "w") as f:
                json.dump(semi_diff, f, indent=4)

            sorted_diff = sorted(semi_diff.items(), key=lambda x: x[1])
            smallest_10 = sorted_diff[:10]
            biggest_10 = sorted_diff[-10:]

            print(f"repredict: {len(sorted_diff)}")
            print("Smallest 10 differences:")
            for idx, diff in smallest_10:
                print(f"Index: {idx}, Difference: {diff}")

            print("\nBiggest 10 differences:")
            for idx, diff in biggest_10:
                print(f"Index: {idx}, Difference: {diff}")
        with open("normalize.json", "w") as f:
            json.dump(normalized, f, indent=4)

        with open("./DAVE_3_shot" + "_" + split + ".json", "w") as f:
            json.dump(predictions, f)

    # with open("big_images.json", "w") as f:
    #    json.dump(bigs, f, indent=4)
    # with open("masked_images.json", "w") as f:
    #    json.dump(masked, f, indent=4)

    plot_queue.put(None)
    plot_process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DAVE", parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)
