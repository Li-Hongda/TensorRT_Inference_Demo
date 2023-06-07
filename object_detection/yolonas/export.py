import argparse
import onnx
import onnxsim
import torch
import torch.nn as nn
from super_gradients.training import models
from super_gradients.common.object_names import Models



class YOLONAS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):

        output = self.model(input)
        return torch.cat(output, dim=-1)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo_nas_m', 
                        choices=['yolo_nas_s','yolo_nas_m', 'yolo_nas_l'] , 
                        help='model.pt')
    parser.add_argument('--save-model', type=str, default='yolonas-m.onnx', 
                        help='model.onnx')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], 
                        help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 export')
    parser.add_argument('--dynamic', action='store_true', help='dynamic axes')
    parser.add_argument('--simplify', action='store_false', help='simplify model')
    parser.add_argument('--opset', type=int, default=11, help='opset version')
    args = parser.parse_args()
    return args


def main(model,
         save_model, 
         img_size,
         batch_size,
         opset = 11,
         half = False,
         dynamic = False,
         simplify = True):
    model = models.get(model, pretrained_weights="coco")
    model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

    model = YOLONAS(model)
    model.eval()
    if dynamic:
        dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}, 
                   'output0': {0: 'batch', 1: 'anchors'}}

    img_size *= 2 if len(img_size) == 1 else 1
    dummy_input = torch.zeros(batch_size, 3, *img_size)

    torch.onnx.export(model, 
                    dummy_input, 
                    save_model, 
                    input_names=['images'],
                    output_names=['output0'],
                    opset_version=opset, 
                    do_constant_folding=True,
                    dynamic_axes=dynamic or None)
    model_onnx = onnx.load(save_model)
    onnx.checker.check_model(model_onnx)
    if simplify:
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, 'simplify failed'
        onnx.save(model_onnx, save_model)

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))