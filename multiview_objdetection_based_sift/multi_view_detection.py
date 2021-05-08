from sift import *
from scipy.ndimage import filters
from numpy import *
from PIL import Image
from pylab import *
from IPython import embed
import glob
import os.path as osp

dir_path = "./data"# views data directory
sift_dir = "./data/sift/"# sift directory

### 数据预处理
def process_dir(dir_path,view_path,view_index):
    # 先获取image所在路径
    # glob.glob()查找符合特定规则的文件路径名
    img_paths = glob.glob(osp.join(dir_path,view_path,'*.jpg'))

    images = []
    gray_images = []
    sifts = []
    i = 0
    for img_path in img_paths:
        i += 1
        images.append(img_path)
        gray_images.append(array(Image.open(img_path).convert('L')))
        # 做sift提取
        sifts.append(process_image(img_path, sift_dir + "view" + str(view_index) + "_" + str(i) + ".sift"))

    return sifts

### 读取特征
def read_features(sifts):
    dists = []
    for sift in sifts:
        # embed()
        l, d = read_features_from_file(sift)
        dists.append(d)

    return dists

### multi-view detect
def match(dists1,dists2):
    num = 0
    for dist1 in dists1:
        for dist2 in dists2:
            matches = match_twosided(dist1, dist2)
            # embed()
            # print(len(matches.nonzero()[0]))
            if (len(matches.nonzero()[0]) > 0):
                num += 1
        # print("-------------------------------")


    print("num is ",num)




if __name__ == "__main__":
    view1_path = "view1"
    view2_path = "view2"

    sifts1 = process_dir(dir_path,view1_path,1)
    sifts2 = process_dir(dir_path,view2_path,2)

    dists1 = read_features(sifts1)
    dists2 = read_features(sifts2)

    match(dists1,dists2)

