"""
Run method pipeline and depth estimation directly for a folder of input images.
"""

import glob, os, re

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
# input images folder
dataset_dir = os.path.join(SCRIPT_PATH,'dataset-sun360')
# Folder to store depth directly on spherical image
depth_dir = os.path.join(SCRIPT_PATH,'sphere-cnn-sun360')
# Folder to store our method's results
results_dir = os.path.join(SCRIPT_PATH, 'results-sun360')

def depthFayao(image, output):
    return not os.system('matlab -nodisplay -nosplash -nodesktop -r -wait "cd %s; demo_modified %s %s; exit;"' %
                         (os.path.join(SCRIPT_PATH, 'depth-fayao', 'demo'), image, output))

def runPipeline(file, output, mainFolder):
    os.system('python run.py -nocrop -noreproject -nodepth -noweighting -i %s -o %s -f %s' % (file, output, mainFolder))

# # uncomment to run depth prediction directly from equirectangular images
# # Run depth prediction on equirectangular images
# depthFayao(dataset_dir, depth_dir)

# Run our mehtod on each image in dataset dir
for file in glob.glob(os.path.join(dataset_dir, "*.jpg")):
    output = os.path.basename(file).replace('.png','')
    runPipeline(file, output, results_dir)
    