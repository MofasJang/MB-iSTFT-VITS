""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from phonemizer import phonemize


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text



def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def chinese_cleaners(text):
  from text.mandarin import chinese_to_ipa
  text = chinese_to_ipa(text)
  text = re.sub(r'\s+$', '', text)
  text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
  return text

def english_cleaners(text):
  '''Pipeline for English text, including abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def english_cleaners2(text):
  '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
  phonemes = collapse_whitespace(phonemes)
  return phonemes

def english_cleaners3(text):
  from text.english import english_to_ipa2
  text = english_to_ipa2(text)
  text = re.sub(r'\s+$', '', text)
  text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
  return text

def engnese_cleaners(text):
  from text.mandarin import chinese_to_ipa
  from text.english import english_to_ipa2
  text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                lambda x: chinese_to_ipa(x.group(1))+' ', text)
  text = re.sub(r'\[EN\](.*?)\[EN\]',
                lambda x: english_to_ipa2(x.group(1))+' ', text)
  text = re.sub(r'\s+$', '', text)
  text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
  return text

def engnese_cleaners2(text):
  from text.mandarin import chinese_to_ipa
  # from text.english import english_to_ipa2
  text = re.sub(r'\[ZH\](.*?)\[ZH\]',
                lambda x: chinese_to_ipa(x.group(1))+' ', text)
  text = re.sub(r'\[EN\](.*?)\[EN\]',
                lambda x: english_cleaners2(x.group(1))+' ', text)
  text = re.sub(r'\s+$', '', text)
  text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
  return text

def japanese_to_ipa(text):
  from text.japanese import japanese_to_ipa2
  text=japanese_to_ipa2(text)
  text = re.sub(r'([^\.,!\?\-…~\\])$', r'\1.', text)
  return text

if __name__=="__main__":
  print("转换前：你好啊我的朋友hellow my friend")
  # print("转换后："+english_cleaners2("你好啊我的朋友hellow my friend"))
  print("转换后："+engnese_cleaners2("[ZH]你好啊我的朋友[ZH][EN]hellow my friend[EN]"))

  