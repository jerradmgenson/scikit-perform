"""
Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the BSD-2-Clause license.
If a copy of the BSD-2-Clause license was not distributed with this
file, You can obtain one at https://opensource.org/licenses/BSD-2-Clause.

"""

import os
import gzip
import shutil
import tempfile
import argparse
from time import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import wget

import benchmarks

BENCHMARKS = {
    'locally linear embedding': (benchmarks.locally_linear_embedding, False),
    'random forest': (benchmarks.random_forest, True),
    'support vector machine': (benchmarks.support_vector_machine, True),
    'xml parsing': (benchmarks.xml_parsing, False),
    'lzma': (benchmarks.lzma_compression, False),
    'sha512': (benchmarks.sha512, False),
    'boyer-moore/horspool': (benchmarks.hamlet_word_count, True),
}


SINGLE_CORE_REFERENCE = 17.156576803752355
MULTI_CORE_REFERENCE = 33.75305533409119


def main():
    cl_args = parse_command_line()
    tmp_dir = create_tmp_dir()
    single_core_scores = []
    multi_core_scores = []
    for name, descriptor in BENCHMARKS.items():
        print(name, '(single-core): ', end='', flush=True)
        seconds = run_test(descriptor[0], 1, tmp_dir,
                           download_data=cl_args.download_data)

        print(str(seconds), 'seconds')
        single_core_scores.append(seconds)
        if descriptor[1] and cpu_count() > 1:
            print(name, '(multi-core): ', end='', flush=True)
            seconds = run_test(descriptor[0], cpu_count(), tmp_dir,
                               download_data=cl_args.download_data)

            print(str(seconds), 'seconds')
            multi_core_scores.append(seconds)

    if cl_args.download_data:
        return 0

    if not cl_args.keep_data:
        shutil.rmtree(tmp_dir)

    single_core_raw_score = sum(single_core_scores) / len(single_core_scores)
    single_core_score = SINGLE_CORE_REFERENCE / single_core_raw_score * 1000
    print('\nsingle core score: {}'.format(int(round(single_core_score))))
    if cpu_count() > 1:
        multi_core_raw_score = sum(multi_core_scores) / len(multi_core_scores)
        multi_core_score = MULTI_CORE_REFERENCE / multi_core_raw_score * 1000
        print('multi core score: {}'.format(int(round(multi_core_score))))


def download_test_data(urls, tmp_dir):
    test_data = {}
    for name, url in urls.items():
        tmp_path = tmp_dir / name
        if not tmp_path.exists():
            wget.download(url, out=str(tmp_path), bar=None)

        if url.endswith('.gz'):
            open_func = gzip.open

        else:
            open_func = open

        with open_func(tmp_path) as fp:
            test_data[name] = fp.read()

    return test_data


def run_test(f, ncores, tmp_dir, download_data=False, *args, **kwargs):
    if hasattr(f, '_data_urls'):
        test_data = download_test_data(f._data_urls, tmp_dir)
        kwargs.update(test_data)

    if download_data:
        return -1

    elif ncores > 1:
        with Pool(ncores) as pool:
            tick = time()
            f(ncores, pool.map, *args, **kwargs)
            tock = time()

    else:
        tick = time()
        f(ncores,
          lambda f, xlist: [f(x) for x in xlist],
          *args,
          **kwargs)

        tock = time()

    return tock - tick


def create_tmp_dir():
    tmp_dir = Path('.').resolve(strict=True) / '.tmp'
    tmp_dir.mkdir(exist_ok=True)

    return tmp_dir


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download-data', action='store_true')
    parser.add_argument('-k', '--keep-data', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main()
