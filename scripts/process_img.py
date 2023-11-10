import sys
sys.path.append('..')
from document_processing import Pipeline
from pathlib import Path
import pprint
import argparse


def process_img(**kwargs):
    """Process 1 image"""

    assert kwargs.get('img_path') is not None, "Missing path to image"
    img_path = kwargs.get('img_path')
    img_size = kwargs.get('img_size')
    check_q = kwargs.get('check_quality')

    pipeline = Pipeline(model_format=kwargs['format'], device=kwargs['device'])
    result = pipeline(img_path, check_quality=check_q, img_size=img_size)
    pp = pprint.PrettyPrinter(depth=4, indent=4)
    pp.pprint(result.full_report)


def main():
    parser = argparse.ArgumentParser(description='Benchmark pipeline')
    parser.add_argument('-i', '--img_path', help='Image path', type=Path, default='../samples/DL_2011/1_CR_DL_2010.jpg', )
    parser.add_argument('-f', '--format', help='Select model format TFlite, ONNX, OpenVINO', type=str,
                        default='OpenVINO')
    parser.add_argument('-d', '--device', help='On which device to run - cpu or gpu', default='cpu', type=str)
    parser.add_argument('--check_quality',
                        help='Is there need to check quality?',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        type=bool)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    args = parser.parse_args()
    params = vars(args)
    process_img(**params)


if __name__ == '__main__':
    main()






