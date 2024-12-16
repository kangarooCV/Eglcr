import sys
sys.path.append('')
import os
import glob
from tqdm import tqdm
from PIL import Image
import os
import cv2
from core.utils import frame_utils




def generate_edge(path):

    disp_list = sorted(glob.glob(path, recursive=True))
    for i, disp_path in enumerate(tqdm(disp_list, ncols=80, ascii=' =')):

        disp = frame_utils.read_gen(disp_path)
        disp = np.array(disp).astype(np.float32)

        mask = disp > 1000
        disp[mask] = 0
        disp[disp == np.inf] = 0

        disp_ = 255.0 * (disp - disp.min()) / (disp.max() - disp.min())
        md = np.median(disp_[disp_ > 0.1])
        np_disp = np.round(disp_).astype(np.uint8)
        threshold1 = int(max(0, (1.0 - 0.94) * md))
        threshold2 = int(min(255, (1.0 + 0.94) * md))
        edge_flow = cv2.Canny(np_disp, threshold1, threshold2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_flow, connectivity=8, ltype=None)
        result = np.zeros((edge_flow.shape[0], edge_flow.shape[1]), np.uint8)  # 创建个全0的黑背景
        for i in range(1, num_labels):
            mask = labels == i
            if stats[i][4] > 50:
                result[mask] = 255
            else:
                result[mask] = 0

        print(disp.min(), disp.max(), " ", disp_path.replace('.pfm', "disp0_edge.png"))
        cv2.imwrite(disp_path.replace('.pfm', '_edge.png'), edge_flow)




if __name__ == '__main__':
    path = r"SceneFlow/disparity/TRAIN/15mm_focallength/scene_forwards/fast/left/*.pfm"
    generate_edge(path)

