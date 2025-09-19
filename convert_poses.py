import os
import numpy as np

root = '/path/to/data_dir'

start = 0
end = -1
scannet = True
all_scenes = sorted(os.listdir(root))
if end == -1:
    end = len(all_scenes)
errors = []
for scene in all_scenes[start:end]:
    print('Scene: ', scene)
    if os.path.exists(os.path.join(root, scene, 'extrinsics.npy')):
        print('Exists')
        continue
    try:
        if scannet: 
            pose_path = os.path.join(root, scene, 'pose')
            all_poses = os.listdir(pose_path)
        else:
            all_poses = [f for f in os.listdir(os.path.join(root, scene)) if f.endswith('.txt') if not f.startswith('intrinsic')]
            pose_path = os.path.join(root, scene)
    except:
        print('Pose directory not exists, check download')
        errors.append(scene)
        continue
    all_poses.sort(key=lambda p: int(p.split('.')[0]))
    out_poses = []

    for pose in all_poses:
        with open(os.path.join(pose_path, pose), 'r') as f:
            lines = f.readlines()
        matrix = [list(map(float, line.strip().split())) for line in lines]
        pose_matrix = np.array(matrix, dtype=np.float32)
        out_poses.append(pose_matrix)

    out_pose_path = os.path.join(root, scene, 'extrinsics.npy')
    out_poses = np.array(out_poses)
    np.save(out_pose_path, out_poses)
print(errors)