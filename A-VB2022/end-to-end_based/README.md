# End-to-end Learning Baseline

We provide the scripts to replicate the end-to-end baseline results of the A-VB competition using End2You. These scripts have been tested with `python3.8`, and you need to install End2You:

``` bash
$ pip install git+https://github.com/end2you/end2you.git
```

First, create the files that End2You requires by using the `create_e2u_files.py` script. This script takes three arguments: `avb_path`, the path to the A-VB data folder, `task`, the task you want to run, and `save_path`, the path to save the output files. For example,

``` python
$ python create_e2u_files.py --avb_path=/path/to/a-vb/data \
                             --save_path=./e2u_files \
                             --task=high
```

After these files are created, you can run the `run_end2you.sh` bash script that automatically runs End2You. This script requires two arguments: (a) the path to the folder where the End2You files were created in the previous step (`save_path` flag), and (b) the task to run End2You for, one of ['high', 'low', 'culture', 'type']. For example,

``` bash
$ bash run_end2you.sh ./e2u_files high
```

For each task, these steps should be repeated from the beginning. `["high", "two", "culture", "type"]`.
