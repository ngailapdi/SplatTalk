import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import torch.nn as nn
import torch.nn.functional as F


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}"

    start = args.start
    end = args.end



    checkpoint = torch.load(ckpt_path)
    model = Autoencoder()
    model = nn.DataParallel(model).cuda()



    model.load_state_dict(checkpoint)
    model.eval()

    all_scenes = np.asarray(sorted(os.listdir(data_dir)))
    if end == -1:
        end = len(all_scenes)

    print(all_scenes)
    for ind, scene in enumerate(all_scenes[start:end]):
        print('Processing id: {}, scene: {}'.format(str(ind), scene))

        train_dataset = Autoencoder_dataset(data_dir, indices=np.array([ind+start]))

        test_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=64,
            shuffle=False, 
            num_workers=16, 
            drop_last=False   
        )


        features = []
        eval_loss = 0.0
        eval_l2loss = 0.0
        eval_cosloss = 0.0
        eval_std = 0.0
        count = 0

        for idx, feature in tqdm(enumerate(test_loader)):
            data = feature.cuda()
            with torch.no_grad():
                outputs = model.module.encode(data)
                pred = model.module.decode(outputs)

            feat_shape = outputs.shape[-1]
            features.append(outputs.cpu())
            mean_norm = torch.norm(data, dim=1).mean()

            l2loss = l2_loss(pred, data)
            cosloss = cos_loss(pred, data)
            loss = l2loss + cosloss
            mean_std = data.std(dim=1).mean(dim=0)
            eval_loss += loss * len(pred)
            eval_l2loss += l2loss * len(pred)
            eval_cosloss += cosloss * len(pred)
            eval_std += mean_std * len(pred)
            count += len(pred)
            del pred
        eval_loss = eval_loss / count
        eval_l2loss = eval_l2loss / count
        eval_cosloss = eval_cosloss / count
        eval_std = eval_std / count
        print("eval_loss: {:.8f}, l2loss: {:.8f}, cossloss: {:.8f}, l2/std: {:.8f}, l2/mean_norm: {:.8f}".format(eval_loss, eval_l2loss, eval_cosloss, eval_l2loss/eval_std, eval_l2loss/mean_norm))

        features = torch.cat(features, dim=0)

        features = features.reshape(-1, 27, 27, feat_shape).permute(0, 3, 1, 2)

        output_dir = os.path.join(data_dir, scene, 'language_feats_256')

        os.makedirs(output_dir, exist_ok=True)
        torch.save(features, os.path.join(output_dir, scene + '.pt'))
        del features
        torch.cuda.empty_cache()