
# ARGPredMix
Antibiotic resistance gene prediction tool based on sequence features and protein language model


**安装 Prodigal：**
```bash
conda install prodigal
```
**安装 Blast：**
```bash
conda install blast
```
**安装Prodigal和blast之前需要添加以下频道：**
```bash
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```
ARGPredMix依靠蛋白质语言模型ESM（https://github.com/facebookresearch/esm）进行特征编码，需要下载ESM2模型到model文件夹下，
下载地址https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt
