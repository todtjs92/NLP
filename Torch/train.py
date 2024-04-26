import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from utils import load_mnist

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt     # x.size = torch.Size([60000, 784])

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)) # torch의 랜덤함수. 크기만큼 랜덤하게 가져옴

    # 이런 함수도 있었군. 텐서가 2개로 나눠들어감 .
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    model = ImageClassifier(28**2, 10).to(device) # 모델 정의해주고
    optimizer = optim.Adam(model.parameters()) # 옵티마이저
    crit = nn.NLLLoss()      # 로스 설정 .

    trainer = Trainer(model, optimizer, crit) # 트레이너를 따로 만들어서 사용 굳이 이럴필요있나??

    trainer.train((x[0], y[0]), (x[1], y[1]), config)
    # 최종적으로 모델 저장까지
    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config,
    }, config.model_fn)

if __name__ == '__main__':
    config = define_argparser() # 이런식으로 config 받으니까 깔끔하네
    main(config)
