## 多语种个性化TTS系统架构

## 一、系统概述

### 1.数据采集

由于模型需要对多语种、多说话人数据进行训练，因此需要收集多个语种的语音集，每个语音集需要包含至少50个不同说话人的音频。

目前完成了对中英文语音集的收集工作，由于其他语种语料库并不完善，因此后续可能需要在视频、音频网站上自行收集并标注语音集。收集的数据集包括中文多说话人85小时语音集[Aishell-3](http://www.aishelltech.com/aishell_3)，英文多说话人585小时语音集[LibriTTS](https://www.openslr.org/60/)，以及用于说话人编码器预训练的英文109说话人语音集[VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)和中文2000说话人语音集[CN-Celeb2](http://www.openslr.org/82/)，还有用于语言编码器预训练的700种语言的多语种语音集[CMU](http://festvox.org/cmu_wilderness/)。

### 2.模型结构

多语种个性化TTS模型结构如下图所示：

<img src="http://pic.panjiangtao.cn/img/image-20230211162419769.png" alt="多语种个性化语音合成模型结构" style="zoom: 67%;" />

#### 2.1 基于VITS的高性能语音合成模型

由韩国的Jaehyeon Kim提出的[Variational Inference with adversarial learning for end-to-end Text-to-Speech（VITS）](https://arxiv.org/pdf/2106.06103)模型实现了高质量语音的合成，该模型结构大致如下图所示：

<img src="http://pic.panjiangtao.cn/img/image-20230211163902553.png" alt="VITS模型结构" style="zoom:67%;" />

VITS的总体结构可以分为5块：

- 先验编码器：由文本编码器生成提升先验分布复杂度的标准化流$f_θ$。应用于多人模型时，向标准化流的残差模块添加说话人嵌入向量。
- 解码器：实际就是声码器HiFi-GAN的生成器。应用于多人模型时，在说话人嵌入向量之后添加一个线性层，拼接到$f_θ$的输出隐变量z。
- 随机时长预测器：从条件输入$h_{text}$估算音素时长的分布。应用于多人模型时，在说话人嵌入向量之后添加一个线性层，拼接到文本编码器的输出$h_{text}$。
- 后验编码器：在训练时输入线性谱，输出隐变量z，推断时隐变量z则由$f_θ$产生。VITS的后验编码器采用WaveGlow和Glow-TTS中的非因果WaveNet残差模块。应用于多人模型时，将说话人嵌入向量添加进残差模块。**（仅用于训练）**
- 判别器：实际就是HiFi-GAN的多周期判别器。**(仅用于训练)**

在该模型的基础上，对其进行下述改进：

1. 由于HiFi-GAN的生成器由大量卷积层组成，因此虽然模型生成的语音质量很高，但是生成速度较慢且模型参数巨大，因此参考[MB-iSTFT-VITS](https://arxiv.org/pdf/2210.15975)中的方法，考虑使用更快速的**逆短时傅里叶变换**和上采样结构代替该解码器，同时通过**多子带生成**来提升合成音频质量。

2. 由于VITS模型为基于GAN的端到端语音合成模型，模型会生成中间变量Mel频谱，将该中间变量替换为**语音后验图(PPG)**作为语言特征，并将其作为编码器的输入。与不仅包含语言信息而且包含丰富声学信息的频谱相比，PPG 传递的信息较少，可以大大简化模型的复杂性。因此引入一个PPG转标准化流的PPG编码器作为**先验编码器**。

   PPG的全称是 phonetic posteriorgrams，即语音后验概率，PPG是一个时间对类别的矩阵，其表示对于一个话语的每个特定时间帧，每个语音类别的后验概率。单个音素的后验概率作为时间的函数称为后验轨迹。

   <img src="http://pic.panjiangtao.cn/img/20210627215421981.png" alt="img" style="zoom:33%;" />

3. 由于单调对齐搜索适合于标准化流与先验序列的对齐，而PPG可以通过PPG编码器生成标准化流，因此只需要对中间变量PPG进行loss计算即可。

#### 2.2 多说话人合成

首先使用经过**预训练的说话人编码器**生成训练音频对应的说话人embedding（spk)，该说话人编码器可以是[d-vector](https://arxiv.org/pdf/)或效果更好的[ECAPA-TDNN](https://arxiv.org/pdf/2005.07143)。

参考[AdaVITS](https://arxiv.org/pdf/2206.00208)中的方法，将说话人embedding融合进**先验编码器**和**iSTFT解码器**中，使模型能够更好地学习每个说话人的音频特征。

#### 2.3 多语种语音合成

首先使用经过**预训练的语言编码器**生成训练音频对应的语言embedding（lang)，该语言编码器可以是d-vector或效果更好的ECAPA-TDNN。

参照谷歌提出的[多语种多说话人模型](https://arxiv.org/pdf/1907.04448.pdf)融合说话人embedding和语言embedding的模型结构，我不仅在解码器中融入这两个特征，也在先验编码器中融入，使模型学习到更丰富的说话人声学特征和语言特征。

<img src="http://pic.panjiangtao.cn/img/image-20230213180737177.png" alt="multispeaker, multilingual text-to-speech" style="zoom:67%;" />

与说话人embedding类似，将语言embedding融合进**先验编码器**和**iSTFT解码器**中。

多语种语音合成需要对文本进行前端预处理，即将不同语言的文本转换为统一格式的音素序列。目前使用最为广泛的方法是使用国际标准音标集IPA进行音素表示。

### 3.模型部署

使用网络剪枝和参数量化技术对模型进行压缩，并使用onnx或torchscript格式进行模型导出，即可在安卓、linux等平台进行快速离线部署，或通过web服务进行在线api部署。

## 二、系统采用策略

### 1.模型的持续学习

多语种TTS旨在在给定相应输入文本的情况下合成不同语言的语音。传统的多语言TTS系统通常需要每种语言的独立模型。最近，端到端多语种TTS系统（即，所有语言的一个模型）已经具有较高性能。这些系统显著降低了部署复杂性，越来越适合实际使用场景。

本系统目前只支持中英文语音合成，而如果后续要添加新的语种，可以通过直接使用新语种语料对原始模型进行微调，这样做的效果可能并不好；也可以选择使用新的语言数据和原始数据从头开始重新训练TTS模型，或者开发联合训练策略来微调原始模型，但这样使得模型训练成本持续提高，因此我选择了持续学习的方法使模型对新语种语料进行微调，防止遗忘策略参考了[Towards lifelong learning of multilingual text-to-speech synthesis](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746968)。


## Multi-band iSTFT VITS and multi-stream iSTFT VITS 
This repository is based on **[official VITS code](https://github.com/jaywalnut310/vits.git)**.<br>
You can train the iSTFT-VITS, multi-band iSTFT VITS (MB-iSTFT-VITS), and multi-stream iSTFT VITS (MS-iSTFT-VITS) using this repository.<br>
We also provide the [pretrained models](https://drive.google.com/drive/folders/1CKSRFUHMsnOl0jxxJVCeMzyYjaM98aI2?usp=sharing).
### 1. Pre-requisites

0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/), then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

### 2. Setting json file in [configs](configs)

| Model | How to set up json file in [configs](configs) | Sample of json file configuration|
| :---: | :---: | :---: |
| iSTFT-VITS | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | ljs_istft_vits.json |
| MB-iSTFT-VITS | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_mb_istft_vits.json |
| MS-iSTFT-VITS | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ljs_ms_istft_vits.json |

### 3. Training
In the case of MB-iSTFT-VITS training, run the following script
```sh
python train_latest.py -c configs/ljs_baker_mini_mb_istft_vits.json -m ljs_baker_mini_mb_istft_vits

```
multispeaker:
```sh
python train_ms.py -c configs/ljs_baker_ms_mini_mb_istft_vits.json -m ljs_baker_ms_mini_mb_istft_vits
```

After the training, you can check inference audio using [inference.ipynb](inference.ipynb)

## References
- https://github.com/jaywalnut310/vits.git
- https://github.com/rishikksh20/iSTFTNet-pytorch.git
- https://github.com/rishikksh20/melgan.git
