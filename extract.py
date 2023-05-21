import torch
import internal.models as models
import internal.dataLoader.volumetricData as dl
import numpy as np
import sys
import zipfile
import glob
import json
import os
import shutil
import time


def getPaddedDimensions(dims,mini_size):
    padding = [(0, (mini_size - dims[i] % mini_size) % mini_size) for i in range(3)]
    padded_dims = [dims[i] + padding[i][1] for i in range(3)]
    return padded_dims

def zip_folder_no_compression(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) < 3:
        print("Usage: python extract.py <source_file> <destination_dir>")
        exit()
    else:
        source_file = sys.argv[1]
        destination_dir = sys.argv[2]

    model = models.Unet(4)
    model.load_state_dict(torch.load('trained/model13.pth'))
    model.eval()

    mini_size = 32

    tmpPath = destination_dir + "tmp"
    with zipfile.ZipFile(source_file, "r") as zip_file:
        zip_file.extractall(tmpPath)

    with open(tmpPath+"/manifest.json",'r') as manifestFile:
        manifest = json.load(manifestFile)
    mblocks = manifest['blocks']
    pt_files = glob.glob(tmpPath + '/blocks/default/*.pt')
    blocks_path = tmpPath + '/blocks/default/'
    for r in range(len(pt_files)):
        filename = blocks_path+"block-"+str(r)+".pt"
        odims = [mblocks[r]['dimensions']['width'],mblocks[r]['dimensions']['height'],mblocks[r]['dimensions']['depth']]
        pdims = getPaddedDimensions(odims,mini_size)
        a = torch.load(filename)
        rs,sk1,sk2 = a
        res = model.forward(rs, mode="extract", q_sk1=sk1, q_sk2=sk2)

        res = res.squeeze(1).detach().numpy()
        res = res.reshape(pdims)
        res = dl.unPadVolume(res, odims)
        res = dl.deNormalizeVolume(res)
        res = dl.toVector(res).astype(np.uint8)
        os.remove(filename)
        with open(blocks_path + os.path.splitext(os.path.basename(filename))[0]+".raw", 'wb') as f:
            f.write(res.tobytes())

    source_file_name = os.path.basename(source_file)
    zip_folder_no_compression(tmpPath, destination_dir + os.path.splitext(source_file_name)[0])
    shutil.rmtree(tmpPath)

    print("--- %s seconds ---" % (time.time() - start_time))



