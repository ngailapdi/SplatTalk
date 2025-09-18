import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, subsample=0, indices=[], feat_dir='language_feats_ov'):
        all_scenes = np.asarray(sorted(os.listdir(data_dir)))
        if len(indices) == 0:
            indices = np.arange(len(all_scenes))

        self.subsample = subsample

        data_path = []
        all_feats = []
        for id, scene in enumerate(all_scenes[indices]):
            print(id, scene)   
            scene_path = os.path.join(data_dir, scene, feat_dir, scene + '.pt')
            data_path.append(scene_path)
            feat = torch.load(scene_path)

            if subsample > 0:
                id = torch.randperm(len(feat))[:subsample]
                feat = feat[id]
            C = feat.shape[1]
            if len(feat.shape) == 4:
                feat = feat.float().permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            all_feats.append(feat)
            del feat
        self.data = torch.cat(all_feats, dim=0)

    def __getitem__(self, index):
        return self.data[index]
        # scene_path = self.data[index//self.subsample]
        # scene = scene_path.split('/')[-1].split('.')[0]
        # if scene in self.all_feats:
        #     return self.all_feats[scene][index % self.subsample]
        # feat = torch.load(scene_path)

        # if self.subsample > 0:
        #     id = torch.randperm(len(feat))[:self.subsample]
        #     feat = feat[id]
        # C = feat.shape[1]
        # if len(feat.shape) == 4:
        #     feat = feat.float().permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        # self.all_feats[scene] = feat
        # # self.data = torch.cat(all_feats, dim=0)
        # return feat[index % self.subsample]
    
    def __len__(self):
        return len(self.data)

class Autoencoder_dataset_feat(Dataset):
    def __init__(self, data_dir, subsample=0, indices=[], feat_dir='language_feats_ov'):
        all_scenes = np.asarray(sorted([f for f in os.listdir(data_dir) if f.startswith('scene')]))
        if len(indices) == 0:
            indices = np.arange(len(all_scenes))

        all_feats = []
        all_points = []
        all_covariances = []
        all_opacitites = []
        for scene in all_scenes[indices]:   
            scene_path = os.path.join(data_dir, scene, feat_dir, scene + '.pt')
            feat1 = torch.load(scene_path)
            if type(feat1) is dict:
                # import pdb; pdb.set_trace()
                feat = feat1['features']
                point = feat1['points']
                covariances = feat1['covariances']
                opacities = feat1['opacitites']
            else:
                feat = feat1
            if subsample > 0:
                id = torch.randperm(len(feat))[:subsample]
                feat = feat[id]
            C = feat.shape[1]
            if len(feat.shape) == 4:
                feat = feat.float().permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            if not type(feat1) is dict:
                point = torch.zeros(len(feat)).half()
                covariances = torch.zeros(len(feat)).half()
                opacities = torch.zeros(len(feat)).half()

                print(point.shape)
            all_feats.append(feat)
            all_points.append(point)
            all_covariances.append(covariances)
            all_opacitites.append(opacities)
        self.data = torch.cat(all_feats, dim=0)
        self.point = torch.cat(all_points, dim=0)
        self.covariances =  torch.cat(all_covariances, dim=0)
        self.opacities = torch.cat(all_opacitites, dim=0)
        print(self.point.shape)


    def __getitem__(self, index):
        return self.data[index], self.point[index], self.covariances[index], self.opacities[index]
    
    def __len__(self):
        return self.data.shape[0]

