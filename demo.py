from torch.nn import DataParallel
from models.dave import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
import os
import matplotlib.patches as patches
from PIL import Image

from utils.data import resize
import matplotlib.pyplot as plt

bounding_boxes = []


def on_click(event):
    # Record the starting point of the bounding box
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # Connect the release event
    fig.canvas.mpl_connect("button_release_event", on_release)


def on_release(event):
    # Record the ending point of the bounding box
    global ix, iy
    x, y = event.xdata, event.ydata
    # Calculate the width and height of the bounding box
    width = abs(x - ix)
    height = abs(y - iy)
    if ix > x :
        ix = x
    if iy > y :
        iy = y
    # Add a rectangle patch to the axes
    rect = patches.Rectangle((ix, iy), width, height, edgecolor="r", facecolor="none")
    ax.add_patch(rect)
    # Store the bounding box coordinates
    bounding_boxes.append((ix, iy, ix + width, iy + height))
    plt.draw()


@torch.no_grad()
def demo(args):
    img_path = "./material/741.jpg"
    global fig, ax
    print("img_path",img_path)
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu
    )
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, "DAVE_3_shot.pth"))["model"],
        strict=False,
    )
    bbox_weight = "/project/g/r13922043/dave_model/detection_3/DAVE_3_shot_4.pth"
    print("bbox_weight",bbox_weight)
    model_state = torch.load(bbox_weight)
    pretrained_dict_box_predictor = {}

    for name, param in model_state["box_predictor"].items():
        pretrained_dict_box_predictor[name.split("box_predictor.")[1]] = param

    model.module.box_predictor.load_state_dict(pretrained_dict_box_predictor,strict=False)

    verification_path = "/project/g/r13922043/dave_model/verification.pth"
    pretrained_dict_feat = {
        k.split("feat_comp.")[1]: v for k, v in torch.load(verification_path)[ "model"].items() if "feat_comp" in k
    }
    pretrained_dict_bbox = {
        k.split("bbox_network.")[1]: v for k, v in torch.load(verification_path)["model"].items() if "bbox_network" in k
    }
    print("Verification model path : ",verification_path)
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    model.module.bbox_network.load_state_dict(pretrained_dict_bbox)
    model.eval()

    image = Image.open(img_path).convert("RGB")
    #image = image.resize((512, 512), Image.LANCZOS)
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Connect the click event
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.title("Click and drag to draw bboxes, then close window")
    # Show the image
    plt.show()

    bboxes = torch.tensor(bounding_boxes)

    img, bboxes, scale = resize(image, bboxes)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    denisty_map, _, tblr, predicted_bboxes = model(img, bboxes=bboxes, evaluation=True)

    plt.clf()
    plt.imshow(image)
    pred_boxes = predicted_bboxes.box.cpu() / torch.tensor(
        [scale[0], scale[1], scale[0], scale[1]]
    )
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]

        plt.plot(  # x0 y0 x1 y1
            [box[0], box[0], box[2], box[2], box[0]],
            [box[1], box[3], box[3], box[1], box[1]],
            linewidth=2,
            color="red",
        )
    plt.title(
        "Dmap count:"
        + str(round(denisty_map.sum().item(), 1))
        + " Box count:"
        + str(len(pred_boxes))
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DAVE", parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
