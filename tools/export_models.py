import argparse
import sys
import os
import torch
import onnx
from pathlib import Path

sys.path.append('../')
from classification.tools import select_device, load_model, \
    check_input_size


def export_models(model,
                  inputs,
                  pts_file,
                  onnx_file):
    # Inference once before export
    outputs = model(inputs)
    print('Inputs shape', str(inputs.shape))
    print('Outputs shape', str(outputs.shape))
    # Export TorchScript and Onnx
    try:
        print('Starting TorchScript export ...')
        torch.jit.trace(model, inputs).save(pts_file)
        print('TorchScript export saved to %s' % pts_file)
    except Exception as err:
        print('TorchScript export failure %s' % err)
    try:
        print('Starting Onnx export ...')
        torch.onnx.export(model,
                          inputs,
                          onnx_file,
                          verbose=False,
                          opset_version=12,
                          input_names=['inputs'],
                          output_names=['outputs'])
        # Check onnx model
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        print('Onnx export saved to %s' % onnx_file)
    except Exception as err:
        print('Onnx export failure %s' % err)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='total batch size')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='model input channels')
    parser.add_argument('--input_size', type=int, default=32,
                        help='model input size')
    parser.add_argument('--save_path', type=str, default='../results',
                        help='path to outputs')
    opt = parser.parse_args()

    device = select_device(opt.device)
    print('Loading model from %s ...' % opt.weights)
    model = load_model(opt.weights, device)
    # Dummy random inputs
    opt.input_size = check_input_size(opt.input_size,
                                      model.max_stride)
    inputs = torch.randn((opt.batch_size,
                          opt.in_channels,
                          opt.input_size,
                          opt.input_size), device=device)
    pt_script_file = os.path.join(opt.save_path,
                                  Path(opt.weights).stem + '.pts')
    onnx_file = pt_script_file.replace('.pts', '.onnx')
    export_models(model,
                  inputs,
                  pts_file=pt_script_file,
                  onnx_file=onnx_file)
