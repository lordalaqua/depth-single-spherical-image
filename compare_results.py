import os, glob, math
import numpy as np
from scipy.io import loadmat, savemat
import cv2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(SCRIPT_PATH,'dataset')

# Read ground truth
def ground_truth(name):
    return os.path.join(SCRIPT_PATH, 'dataset', 'ground-truth','%s.exr' % name)

def results(name):
    return os.path.join(SCRIPT_PATH, 'results_90', name, 'depth.mat')

def sphere_cnn(name):
    return os.path.join(SCRIPT_PATH, 'sphere-cnn', name, 'predict_depth.mat')

ncc_mean_ours = 0
ncc_mean_direct = 0
i = 0
for file in glob.glob(os.path.join(dataset_dir, "*.png")):
    name = os.path.basename(file).replace('.png','')
    direct = loadmat(sphere_cnn(name))['data_obj']
    ours = cv2.resize(loadmat(results(name))['data_obj']*50, direct.shape, cv2.INTER_AREA)
    ground_t = cv2.resize(cv2.imread(ground_truth(name), cv2.IMREAD_UNCHANGED)[...,0], direct.shape, cv2.INTER_AREA)
    direct = direct.T
    # Flatten images to 1D array for comparing
    # Use mask to get only pixels where depth was estimated in our method
    im = ours.ravel()
    mask = im > 0
    im = im[mask]
    std_ours = np.std(im)
    mean_ours = np.mean(im)
    sp = direct.ravel()
    sp = sp[mask]
    std_direct = np.std(sp)
    mean_direct = np.mean(sp)
    gt = ground_t.ravel()
    gt = gt[mask]
    std_gt = np.std(gt)
    mean_gt = np.mean(gt)
    n = float(len(im))
    ncc_sum_ours = 0
    ncc_sum_direct = 0
    # Calculate sums over arrays
    for est_i, dir_i, gt_i in zip(im, sp, gt):
        gt_term_mean = (gt_i - mean_gt)
        ncc_sum_ours += (est_i - mean_ours)*gt_term_mean
        ncc_sum_direct += (dir_i - mean_direct)*gt_term_mean
    ncc_ours = (1/n) * 1/(std_ours*std_gt) * ncc_sum_ours
    ncc_direct = (1/n) * 1/(std_direct*std_gt) * ncc_sum_direct
    ncc_mean_ours += ncc_ours
    ncc_mean_direct += ncc_direct
    i += 1
    # Print in table-like format for latex
    print("%s & %.5f & %.5f & %.5f \\\\"%(name, ncc_ours, ncc_direct, ncc_ours-ncc_direct))
ncc_mean_ours /= i
ncc_mean_direct /= i
print("mean & %.5f & %.5f & %.5f"%(ncc_mean_ours,ncc_mean_direct, ncc_mean_ours-ncc_mean_direct))
