import argparse
import sys
sys.path.append('../')
import os
import torch
import struct
from pathlib import Path
from classification.tools import load_model


def export_weights(model, wts_file):
    # Export model to TensorRT compatible format
    with open(wts_file, 'w') as fd:
        fd.write('{}\n'.format(len(model.state_dict().keys())))
        for key, val in model.state_dict().items():
            vec = val.reshape(-1).cpu().numpy()
            fd.write('{} {} '.format(key, len(vec)))
            for x in vec:
                fd.write(' ')
                fd.write(struct.pack('>f', float(x)).hex())
            fd.write('\n')
    print('Export done, weights saved to %s' % wts_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--save_path', type=str, default='../results',
                        help='path to outputs')
    opt = parser.parse_args()

    device = torch.device("cpu")
    print('Loading model from %s ...' % opt.weights)
    model = load_model(opt.weights, device)
    wts_file = os.path.join(opt.save_path,
                            Path(opt.weights).stem + '.wts')
    export_weights(model, wts_file)
