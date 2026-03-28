import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_img_with_boxes(img: torch.Tensor, bboxes: torch.Tensor):
    '''Shows an image with the predicted bounding boxes.'''
    img_array = img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img_array)

    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        
        width = xmax - xmin
        height = ymax - ymin
        
        rect = patches.Rectangle(
            (xmin, ymin), width, height, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()

def intersection_over_union(predicted,original):
    '''Compute Intersection over Union (IoU) for two axis-aligned rectangles.

    The function assumes each box is in [xmin, ymin, xmax, ymax] format.
    Steps:
    1. Find the coordinates of the overlap rectangle: left, top, right, bottom.
    2. Compute intersection area = max(0, right-left) * max(0, bottom-top).
    3. Compute each box area independently.
    4. Compute union area = area_predicted + area_original - intersection_area.
    5. Return IoU = intersection_area / union_area.

    IoU is 0 when boxes do not overlap and 1 when they are identical.
    '''
    xA = max(predicted[0],original[0])
    yA = max(predicted[1],original[1])
    xB = min(predicted[2],original[2])
    yB = min(predicted[3],original[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    predicted_area = (predicted[2]-predicted[0]) * (predicted[3]-predicted[1])
    original_area = (original[2]-original[0]) * (original[3]-original[1])

    union_area = predicted_area + original_area - inter_area

    return inter_area / union_area