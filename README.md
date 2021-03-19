
# README

## How to use.

1. create conda environment

2. install pytorch>=1.7.1, torchvision>=0.8.2

3. run `pip install -r requirements.txt`

4. download this repository to your local machine

5. uncompress and orginaze the NTRIE 2021 NHDehazing data in a folder as following:

```
NTIRE2021NHHAZE:
    test:
        source:
            img_idx_0.png
            img_idx_1.png
            ...
```

6. modify 44th line in `/path/to/root/of/project/src/configs/configs.py`:

`"NTIRE2021NHHAZE": "/path/to/your/dataset"`

for example, if you put at: `/data/NTIRE2021NHHAZE/test/source/`, then the 44th line should be: `"NTIRE2021NHHAZE": "/data/NTIRE2021NHHAZE/test/source/"`.

7. download the pre-trained [parameters](https://mailsdueducn-my.sharepoint.com/:u:/g/personal/201700181055_mail_sdu_edu_cn/EbWH_wGOX3BMoT-CryYcIC8BiAJ2l7p9SzpXGZJHoW5P5A?e=KDYvEB) into `/path/to/root/of/project/checkpoints/Test/`

8. the following command should generate results in folder `/path/to/root/of/project/results/Test/NTIRE2021NHHAZE/results`:

```bash
cd /path/to/root/of/project
python src/main.py --id Test --dataset NTIRE2021NHHAZE --batch_size 1 --resume true --gpu [0]
```
We have uploaded the results on this repository, please check folder `results/Test/NTIRE2021NHHAZE/results`
