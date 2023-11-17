# -*- coding: utf-8 -*-
# @Author   : zqian9
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import ASVSpoof21
from model import resnetv2_18, resnetv2_34, resnetv2_10
from metrics21 import eval_to_score_file_la

# set used GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict(dataloader, model, score_path, device):
    fnames, scores = [], []
    model.eval()
    iterations = len(dataloader)

    with torch.no_grad():
        for i, (batch_x, batch_filename) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            batch_out = F.log_softmax(batch_out, dim=1)
            batch_score = batch_out[:, 0].detach().cpu().numpy().ravel().tolist()

            fnames.extend(batch_filename)
            scores.extend(batch_score)

            print('\rCompleted: {}%'.format(int(i * 100 / iterations)), end="")
        print("\r[INFO] Inference completed")

    if score_path is not None:
        with open(score_path, 'w') as f:
            for filename, score in zip(fnames, scores):
                f.write('{} {}\n'.format(filename, score))
        print("[INFO] Scores save to {}".format(score_path))
    else:
        print("[INFO] Please input score path")


def main():
    test_dataset = ASVSpoof21(
        data_root=config.data_root,
        duration=config.duration,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] Evaluation on device: {}'.format(device.upper()))

    model = resnetv2_10().to(device)
    print('[INFO] Model loaded...')

    model.load_state_dict(torch.load(Path(config.ckpt_root) / "best_acc_model.pt"))
    predict(test_dataloader, model, score_path="./scores.txt", device=device)

    asv_key_file = Path(config.data_root) / Path("LA2021/keys") / Path("ASV/trial_metadata.txt")
    asv_scr_file = Path(config.data_root) / Path("LA2021/keys") / Path("ASV/ASVTorch_Kaldi/score.txt")
    cm_key_file = Path(config.data_root) / Path("LA2021/keys") / Path("CM/trial_metadata.txt")
    min_t_dcf, eer_cm = eval_to_score_file_la(
        "scores.txt", cm_key_file, asv_key_file, asv_scr_file, 'eval')
    print("[INFO] min t-DCF: {:6.4f} - EER: {:5.2f}%".format(min_t_dcf, eer_cm * 100))


if __name__ == '__main__':
    main()
