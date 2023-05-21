import torch
import internal.models as models
import internal.dataLoader.volumetricData as dl
import time
import os
import sys
import shutil
import zipfile


def copy_file(source_file, destination_dir):
    try:
        shutil.copy2(source_file, destination_dir)
    except FileNotFoundError:
        print(f"Error: File '{source_file}' not found.")
    except IsADirectoryError:
        print(f"Error: '{destination_dir}' is a directory. Please provide a valid destination filename.")
    except Exception as e:
        print(f"An error occurred while copying the file: {str(e)}")

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
        print("Usage: python compress.py <source_file> <destination_dir>")
        exit()
    else:
        source_file = sys.argv[1]
        destination_dir = sys.argv[2]
        copy_file(source_file, destination_dir)

    filePath = destination_dir+os.path.basename(source_file)
    blocks,manifest = dl.readBvpFiles(filePath)

    # Load the model and set it to eval mode
    model = models.Unet(4)
    model.load_state_dict(torch.load('trained/model2.pth'))
    model.eval()

    mini_block = 32

    compressed_blocks = []

    # Compress the blocks
    for i in range(len(blocks)):
        # Preprocess the block for compression, each block is of shape (values, dimensions, name)
        # Convert uint8 1D array to a float32 3D array (volume) and normalize it
        block = dl.toVolume(blocks[i][0],blocks[i][1])
        block = dl.normalizeVolume(block)
        # Pad the volume to mini_block size and split it to mini_blocks
        block = dl.padVolume(block,mini_block)
        block = dl.toMiniVolumes(block,mini_block)
        # Convert the batch of mini blocks to a tensor and add an empty dimension indicating 1 channel
        block = torch.from_numpy(block).unsqueeze(1)

        # Compress the block
        rs, sk1, sk2 = model.forward(block, mode="compress")
        compressed_data = [rs, sk1, sk2]
        compressed_blocks.append([compressed_data,blocks[i][2]])

    # Save the compressed blocks to the output bvp file
    tmpPath = destination_dir+"tmp"
    with zipfile.ZipFile(filePath, "r") as zip_file:
        zip_file.extractall(tmpPath)

    shutil.rmtree(tmpPath+"/blocks/default")
    os.makedirs(tmpPath+"/blocks/default", exist_ok=True)

    for l in range(len(compressed_blocks)):
        torch.save(compressed_blocks[l][0],tmpPath + "/blocks/default/"+os.path.splitext(compressed_blocks[l][1])[0]+".pt")

    source_file_name = os.path.basename(source_file)
    zip_folder_no_compression(tmpPath,destination_dir+os.path.splitext(source_file_name)[0]+".bvp.n")

    os.remove(destination_dir+source_file_name)
    shutil.rmtree(tmpPath)

    print("--- %s seconds ---" % (time.time() - start_time))




