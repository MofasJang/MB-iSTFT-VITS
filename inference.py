#coding=utf-8
from distutils.command.config import config
import re
import time
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_model(config_path,checkpoint_path,speaker=None):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = net_g.requires_grad_(False)

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    return net_g.infer

def synthesize(text,model,config_path,speaker=None):
    hps = utils.get_hparams_from_file(config_path)
    start=time.time()
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = model(x_tst, x_tst_lengths, sid=speaker,noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    wavfile.write(os.path.join("sample", "{}.wav".format(re.sub(r'([\\/\:\*\.,!\?\-…~<>|])$',"",text[:20]))), hps.data.sampling_rate, audio)
    print(time.time()-start)
if __name__=="__main__":
    texts=[
            # "医者は活溌にまた無雑作に津田の言葉を否定した。併せて彼の気分をも否定するごとくに。",
            "[ZH]众里寻他千百度,蓦然回首,[ZH][EN]Hey,how do you do[EN]",
            "[ZH]问君能有几多愁,[ZH][EN]as a boy without a girl[EN]",
            # "[ZH]问君能有几多愁,[ZH][EN]easy come easy go[EN]",
            # "[ZH]满园春色关不住,[ZH][EN]Friday is coming soon[EN]",
            # "[ZH]两只黄鹂鸣翠柳,[ZH][EN]what place shall we go?[EN]",
            # "[ZH]江山如此多娇,[ZH][EN]you are so small[EN][ZH]惜秦皇汉武,[ZH][EN]too simple,[EN][ZH]唐宗宋祖[ZH][EN]sometime naieve[EN]",
            # "[ZH]一代天骄,成吉思汗,[ZH][EN]can't play football.[EN][ZH]俱往矣,数风流, [ZH][EN]die all[EN]",
            # "[ZH]但使龙城飞将在,[ZH][EN]come on baby don't be shy[EN]",
            # "[ZH]昨夜西风凋碧树,独上高楼,[ZH][EN]watching Chairman Hu[EN]",
            # "[ZH]两岸猿声啼不住,[ZH][EN]monkey go to sexing zoo[EN]",
            # "[ZH]林花谢了春红,太匆匆,[ZH][EN]where is my iphone?[EN]",
            # "[ZH]身无彩凤双飞翼,[ZH][EN]Get away from me[EN]",
            # "[ZH]春城无处不飞花,[ZH][EN]let's go to the cinema[EN]",
            # "[ZH]山穷水复疑无路,[ZH][EN]how can i find a girl like you?[EN]",
            # "[ZH]路见不平一声吼,[ZH][EN]kick your ass let's go[EN]",
            # "[ZH]天苍苍,野茫茫,[ZH][EN]baby let's have fun![EN]",
            # "[ZH]爆竹声中一岁除,[ZH][EN]never had a dream come true[EN]",
            # "[ZH]人生自是有情痴,[ZH][EN]let's have sex[EN]",
            # "[ZH]花自飘零水自流,[ZH][EN]may i help you?[EN]",
            # "[ZH]十年一觉扬州梦,[ZH][EN]Screw you guys, I'm going home[EN]",
            # "[ZH]遥知兄弟登高处,[ZH][EN]you making love with who?[EN]",
            # "[ZH]茕茕白兔,东游西顾。[ZH][EN]Hey boy, you are fucking cool[EN]",
            # "[ZH]山外青山楼外楼,[ZH][EN]baby I'll rock you[EN]",
            # "[ZH]劝君更尽一杯酒,[ZH][EN]too late to say I love you[EN]",
            # "[ZH]垂死病中惊坐起,[ZH][EN]hey,I want to pee[EN]",
            # "[ZH]古来圣贤皆寂寞,[ZH][EN]what do I study for?[EN]",
           ]
    config_path="configs/baker_ljs_ms.json"
    checkpoint_path="logs/baker_ljs_ms/G_310000.pth"
    model=get_model(config_path,checkpoint_path)
    for text in texts:
        synthesize(text,model,config_path)
    
    # texts=["[ZH]据玛丽亚修女的公开日记所记录。故事的开始是午后的三点钟,拜旦城的光明教堂的一间非正式会客室,卡尔先生和奥古斯都先生以及玛丽亚女士三人在等着教皇保罗的召见,这是惯例。每个月的月1号玛丽亚女士都要来,并非每次都能见到教皇,如果下午5点钟的钟声响起之前,教皇没有出现她便会离开会客室,前往修女的食堂用餐,这也是惯例。玛丽亚坐在靠着窗户的桌子旁,她把这张桌子当成了书桌,桌面上摆着一个长方形圆边的树枝编制成的有盖筐子,筐子里面的东西摆在桌子上了,是一本厚厚的笔记本、彩色铅笔、墨水笔和一瓶墨水,铅笔用来作画,墨水笔用来书写文字。玛丽亚有着很好的记忆里,她喜欢把她的能力用在记录她所见过的植物与动物以及这些植物与动物生活的环境。[ZH]",
        # "英伟达开源的自然语音处理开发套件 from the standpoint of the good of the industries themselves, as well as the general public interest",
        # "from the standpoint of the good of the industries themselves, as well as the general public interest,",
        # "secret service agents formed a cordon to keep the press and photographers from impeding their passage and scanned the crowd for threatening movements.",
        # "especially as no more time is occupied, or cost incurred, in casting, setting, or printing beautiful letters",
        # "the uncle claimed her. the husband resisted.",
        # "they bought their offices from one another, and were thus considered to have a vested interest in them.",
        # "in the center of the chapel was the condemned pew, a large dock-like erection painted black.",
        # "again, a turnkey deposed that his chief did not enter the wards more than once a fortnight.",
        # "while neglecting to maintain his unity of ideal in the case of nearly all the numerous species of snakes, he should have added a tiny rudiment in the case of the python",
        # "the department hopes to design a practical system which will fully meet the needs of the protective research section of the secret service.",
        # ]
    # config_path="./configs/baker_ljs_ms.json"
    # checkpoint_path="./logs/baker_ljs_ms/G_310000.pth"
    # model=get_model(config_path,checkpoint_path)
    # for text in texts:
    #     synthesize(text,model,config_path,checkpoint_path)