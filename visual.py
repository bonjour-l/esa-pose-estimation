import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from transforms3d.euler import euler2mat
#from skimage.io import imsave

def visualize_bounding_box(rgb, corners_pred, corners_targets=None, centers_pred=None, centers_targets=None, save=False, save_fn=None):
    '''

    :param rgb:             torch tensor with size [b,3,h,w] or numpy array with size [b,h,w,3]
    :param corners_pred:    [b,1,8,2]
    :param corners_targets: [b,1,8,2] or None
    :param centers_pred:    [b,1,2] or None
    :param centers_targets:  [b,1,2] or None
    :param save:
    :param save_fn:
    :return:
    '''
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(0, 2, 3, 1).detach().cpu().numpy()
    rgb = rgb.astype(np.uint8)

    batch_size = corners_pred.shape[0]
    for idx in range(batch_size):
        fig, ax = plt.subplots(1)
        ax.imshow(rgb)
        if corners_targets is not None:
            ax.add_patch(patches.Polygon(xy=corners_targets[idx, 0][[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1,
                                         edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corners_targets[idx, 0][[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1,
                                         edgecolor='g'))
        ax.add_patch(
            patches.Polygon(xy=corners_pred[idx, 0][[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(
            patches.Polygon(xy=corners_pred[idx, 0][[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))

        if centers_pred is not None:
            ax.plot(centers_pred[idx, 0, 0],centers_pred[idx, 0, 1],'*')
        if centers_targets is not None:
            ax.plot(centers_targets[idx, 0, 0], centers_pred[idx, 0, 1], '*')

        plt.axis("off")

        height, width, channels = rgb.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width/100.0/4.0, height/100.0/4.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)

        if not save:
            plt.show()
        else:
            plt.savefig(save_fn.format(idx))
        plt.close()