import numpy as np
import internal.models as models
import torch
import internal.dataLoader.volumetricData as dl
import torchmetrics as tm

def loss(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    avg_diff = np.mean(diff)
    return avg_diff

if __name__ == '__main__':
    mini_size = 32
    model = models.Unet(4)
    model.load_state_dict(torch.load('trained/model13.pth'))
    model.eval()

    use_batching = True
    files,manifest = dl.readBvpFiles(folderPath="volumes/turbulence/turbulence.bvp")
    avg_loss = 0
    avg_psnr = 0
    for k in range(len(files)):
        odims = np.array(files[k][1])
        ofile = files[k][0]
        ffile = ofile.astype(np.float32)
        vol = dl.toVolume(ofile,odims).astype(np.float32) / 255
        vol = dl.padVolume(vol,mini_size)
        pdims = vol.shape
        vols = dl.toMiniVolumes(vol,mini_size)
        vdims = vols.shape
        vols = torch.from_numpy(vols).unsqueeze(1)
        res = torch.empty(vdims)
        org = res.unsqueeze(1)
        test_save = []
        if use_batching:
            rs, sk1, sk2 = model.forward(vols, mode="compress")
            rs = model.forward(rs, mode="extract", q_sk1=sk1, q_sk2=sk2)
            res = rs
        else:
            for d in range(vdims[0]):
                inp = vols[d].unsqueeze(0)
                rs,sk1,sk2 = model.forward(inp,mode="compress")
                saved = [rs, sk1, sk2]
                test_save.append(saved)
                rs = model.forward(rs,mode="extract",q_sk1=sk1,q_sk2=sk2)
                res[d] = rs

        res = res.detach().numpy()
        res = res.reshape(pdims)
        res = dl.unPadVolume(res,odims)
        out = np.abs(res.reshape(-1) *255).round()
        avg_loss += loss(ffile,out)
        out = out.astype(np.uint8)
        #filename = "output/block-"+str(k)+".raw"
        #with open(filename, 'wb') as f:
        #    f.write(out.tobytes())
        psnr = tm.PeakSignalNoiseRatio(data_range=255.0)
        avg_psnr += psnr(torch.tensor(out), torch.tensor(files[k][0]))

    print("total MAE loss")
    print(avg_loss/len(files))
    print("psnr")
    print(avg_psnr / len(files))


