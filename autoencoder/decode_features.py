import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset, Autoencoder_dataset_feat
from model import Autoencoder
import torch.nn as nn
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_feat_dir', type=str, required=True)
    parser.add_argument('--overwrite', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)



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
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}"


    checkpoint = torch.load(ckpt_path)
    model = Autoencoder()
    model = nn.DataParallel(model).cuda()



    model.load_state_dict(checkpoint)
    model.eval()

    all_scenes = np.asarray(sorted([f for f in os.listdir(data_dir) if f.startswith('scene')]))

    print(all_scenes)
    for ind, scene in enumerate(all_scenes):
        print('Processing id: {}, scene: {}'.format(str(ind), scene))
        if not args.overwrite:
            if os.path.exists(os.path.join(data_dir, scene, args.output_feat_dir)):
                print('Exists: ', scene)
                continue

        if not args.render:
            dataset = Autoencoder_dataset_feat(data_dir, indices=np.array([ind]), feat_dir='feat_fs')
        else:
            dataset = Autoencoder_dataset_feat(data_dir, indices=np.array([ind]), feat_dir='')


        test_loader = DataLoader(
            dataset=dataset, 
            batch_size=256,
            shuffle=False, 
            num_workers=16, 
            drop_last=False   
        )


        features = []
        points = []
        covariances = []
        opacities = []



        for idx, (feature, point, covariance, opacity) in tqdm(enumerate(test_loader)):
            data = feature.cuda()
            with torch.no_grad():
                outputs = model.module.decode(data)

            features.append(outputs.cpu())
            points.append(point)
            covariances.append(covariance)
            opacities.append(opacity)


        features = torch.cat(features, dim=0)
        points = torch.cat(points, dim=0)
        covariances = torch.cat(covariances, dim=0)
        opacities = torch.cat(opacities, dim=0)
        print(features.shape)
        feat_shape = features.shape[-1]
        if args.render:
            features = features.reshape(-1, 32, 32, feat_shape).permute(0, 3, 1, 2)

        output_dir = os.path.join(data_dir, scene, args.output_feat_dir)

        os.makedirs(output_dir, exist_ok=True)
        if args.render:
            torch.save(features,  os.path.join(output_dir, scene + '.pt'))
        else:
            torch.save({'features':features, 'points': points, 'covariances': covariances, 'opacities': opacities}, os.path.join(output_dir, scene + '.pt'))
