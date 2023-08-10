def mysum(x,y):
    return x+y

mysum(1,2)

mysum(10,20)

class Calculator(object):
    def __init__(self, base):
        self.base = base

calculator = Calculator(10)

calculator

calculator() # error

class Calculator2(object):
    def __init__(self, base):
        self.base = base
    
    def __call__(self, x, y):
        return self.base + x + y

calc2 = Calculator2(10)

calc2.__call__(1,2)

calc2.__call__(10,20)

calc2(10, 200)

import requests
from bs4 import BeautifulSoup
from collections import Counter

def word_count(url): # 문자열 수집
    html = requests.get(url).text
    # 단어 리스트로 변환
    soup = BeautifulSoup(html, 'html.parser')
    words = soup.text.split()
    # 단어수 카운트
    counter = Counter(words)
    return counter

word_count('https://nomade.kr/vod/')

import re
import requests
from bs4 import BeautifulSoup
from collections import Counter

def korean_word_count(url): # 문자열 수집
    html = requests.get(url).text
    # 단어 리스트로 변환
    soup = BeautifulSoup(html, 'html.parser')
    words = soup.text.split()
    # 한글 단어만 추출
    words = [word for word in words if re.match(r'^[ㄱ-힣]+$', word)] # 이 코드만 추가
    # 단어수 카운트
    counter = Counter(words)
    # 통계 출력
    return counter    

korean_word_count('https://nomade.kr/vod/')

import requests
from bs4 import BeautifulSoup
from collections import Counter

class WordCount(object):
    def get_text(self, url):
        '문자열 수집'
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        return soup.text
    
    def get_list(self, text): 
        '단어 리스트로 변환'
        return text.split()
    
    def __call__(self, url):
        text = self.get_text(url)
        words = self.get_list(text)
        counter = Counter(words)
        return counter

word_count = WordCount()

word_count('https://nomade.kr/vod/')

import re
class KoreanWordCount(WordCount):
    def get_list(self, text):
        words = text.split()
        return [word for word in words if re.match(r'^[ㄱ-힣]+$', word)] #list comprehension 문법

korean_word_count = KoreanWordCount()

korean_word_count('https://nomade.kr/vod/')



