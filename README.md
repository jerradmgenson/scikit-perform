# scikit-perform

scikit-perform is a benchmark suite for scientific computing. It primarily
measures CPU and memory performance. Its main virtues compared to other benchmarks
are that it is non-synthetic - it's based on algorithms and data actually used in the
real world - it is targeted at scientific computing as opposed to more common
benchmark demographics such as gaming, and it is fully cross-platform, both with
respect to the operating system and hardware architecture; any system that has a
BLAS and LAPACK implementation and can run Python 3.x should be able to run
scikit-perform. This makes it useful for benchmarking unusual computing machinery
that may be unable to run conventional benchmarks and comparing them with everyday
laptops and desktops, and even high-end workstations.

## Copyright and License

Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the BSD-2-Clause license.
If a copy of the BSD-2-Clause license was not distributed with this
file, You can obtain one at https://opensource.org/licenses/BSD-2-Clause.

## Running scikit-perform

This steps were written for Ubuntu 20.04, but they should work on other
operating systems with slight modifications.

1. Clone the git repo or download the code to your local computer.
2. Open a terminal and change into the directory containing the code.
3. Run `python3 -m venv .pyenv` to create a virtual environment.
4. Activate the virtual environment with `source .pyenv/bin/activate`.
5. Install the dependencies with `pip install -r requirements.txt`.
6. Close out of all other running applications.
7. Run `python skperform.py` to start the benchmark.

## High scores

make | model | submodel | year | cpu | memory | os | single-core score | multi-core score | notes
---- | ----- | -------- | ---- | --- | ------ | -- | ----------------- | ---------------- | -----
Acer|Aspire|E 15|2018|Intel Core i5-8250U |8GB DDR4 |Linux Mint 20.1 Ulyssa - Cinnamon (64-bit)|999|3025|
HP|EliteBook|830 G6|2019|Intel Core i5-8365U|16GB DDR4-2400 SDRAM|Windows 10 Enterprise|880|1931|
HP|Chromebook 14|ak050nr|2015|Intel Celeron N2940|4GB DDR3L SDRAM|Lubuntu 20.04.2 amd64|308|957|
Radxa|Rock Pi X|v1.4|2020|Intel Atom x5-Z8350 Cherry Trail|2GB LPDDR3@1866Mb/s|Ubuntu Server 20.04.2 amd64|257|733|
Raspberry Pi|4|Model B|2019|Broadcom BCM2711 Quad core Cortex-A72|4GB LPDDR4-3200 SDRAM|Ubuntu Server 20.04.4 arm64|253|698|
TI|BeagleBone Black|Revision C|2014|AM335x 1GHz ARM Cortex-A8|512MB DDR3 RAM|Debian 10|10||
TI|PocketBeagle|Revision A2|2017|Octavo Systems OSD3358-SM 1-GHz ARM Cortex-A8|512MB DDR3 RAM|Debian 10|13||