import sys
sys.path.append('..')
from document_processing import Pipeline
from pathlib import Path
import argparse
import json
import cpuinfo
import pandas as pd
import numpy as np

def benchmark(**kwargs):

    def test_pic(img_folder: Path):
        benchmark_list = []

        for _ in range(kwargs['cicles']):
            for img in img_folder.glob('**/*.*'):
                result = pipeline(img)
                benchmark_list.append(result.timings['total'])

        return np.mean(benchmark_list)


    pipeline = Pipeline(model_format=kwargs['format'], device=kwargs['device'])


    images_folder = Path(kwargs['images'])
    benchmark_folder = Path(kwargs['save_to'])
    benchmark_folder = benchmark_folder.joinpath(f"{kwargs['format']}_{kwargs['device']}.csv")
    doctypes = [folder for folder in images_folder.iterdir() if folder.is_dir()]

    #preheat model
    pipeline(next(iter(images_folder.glob('**/*.jpg'))))


    if benchmark_folder.is_file():
        df_result = pd.read_csv(benchmark_folder, index_col=0)
    else:
        raw_doctypes = list(map(lambda x: x.stem, doctypes))
        df_result = pd.DataFrame(columns=raw_doctypes)

    cpu_name = cpuinfo.get_cpu_info()['brand_raw']

    bench_result = {cpu_name: {}}


    if len(doctypes) > 0:
        for img_folder in doctypes:
            print(f'[*] Collecting info for {img_folder.name}')
            result = test_pic(img_folder)
            bench_result[cpu_name][img_folder.stem] = result
    else:
        result = test_pic(images_folder)
        bench_result[cpu_name]['all'] = result

    benchmark_folder.parent.mkdir(parents=True, exist_ok=True)
    df_result = pd.concat((df_result, pd.DataFrame.from_dict(bench_result, orient='index')), ignore_index=False, axis=0)
    df_result = df_result.groupby(df_result.index).mean()
    df_result = df_result.round(3)
    df_result.to_csv(benchmark_folder)

def main():
    parser = argparse.ArgumentParser(description='Benchmark pipeline')
    parser.add_argument('-i', '--images', help='Where to save results', type=str, default='images')
    parser.add_argument('-s', '--save_to', help='Where to save result in JSON format', type=str,
                        default=r'bench_results')
    parser.add_argument('-f', '--format', help='Select model format TFlite, ONNX, OpenVINO', type=str,
                        default='ONNX')
    parser.add_argument('-d', '--device', help='On which device to run - cpu or gpu', default='cpu', type=str)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    parser.add_argument(
        '--cicles',
        help='How many cicles to run in images, more better accuracy',
        required=False,
        default=5,
        type=int)


    args = parser.parse_args()
    params = vars(args)

    benchmark(**params)


if __name__ == '__main__':
    main()
