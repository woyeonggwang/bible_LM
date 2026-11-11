import re
import json
import time
from collections import Counter

PATTERN = r" ?\w+| ?[^\w\s]+|\s+"

def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)           
            i += 2                      
        else:
            out.append(ids[i])          
            i += 1                     
    return out

def train_bpe(text, num_merges):
    words = Counter(re.findall(PATTERN, text))       
    chars = sorted(set(text))                        
    vocab = {i: ch for i, ch in enumerate(chars)}    
    stoi = {ch: i for i, ch in vocab.items()}        
    splits = {w: [stoi[c] for c in w] for w in words}  
    merges = {}                                      

    for k in range(num_merges):
        stats = Counter()                            
        for w, freq in words.items():
            ids = splits[w]
            for pair in zip(ids, ids[1:]):           
                stats[pair] += freq                  
        if not stats:                                
            break
        best = max(stats, key=stats.get)             
        new_id = len(vocab)                          
        merges[best] = new_id                        
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]  
        for w in words:
            splits[w] = merge(splits[w], best, new_id)   
        if k < 30 or (k + 1) % 100 == 0:             
            print(f'{k + 1}: {vocab[new_id]!r}  ({stats[best]}번)')   
    return merges, vocab

class BPETokenizer:
    def __init__(self, chars, merges):
        self.chars = chars
        self.merges = merges
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        for (a, b), i in merges.items():                 
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self.cache = {} 
    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        ids = [self.stoi[c] for c in word]
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))               
            pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
           
            if pair not in self.merges:                  
                break
            ids = merge(ids, pair, self.merges[pair])   
        self.cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for word in re.findall(PATTERN, text):           
            ids.extend(self.encode_word(word))          
        return ids

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)      

    def save(self, path):
        data = {
            'chars': self.chars,
            'merges': [[a, b, i] for (a, b), i in self.merges.items()],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)       

    @classmethod                                         
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        merges = {(a, b): i for a, b, i in data['merges']}   
        return cls(data['chars'], merges)        
if __name__ == '__main__':
    with open('bible.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    merges, vocab = train_bpe(text, num_merges=500)
    tok = BPETokenizer(sorted(set(text)), merges)        
    tok.save('bpe.json')                                 

    tok = BPETokenizer.load('bpe.json')                  
    t0 = time.time()
    ids = tok.encode(text)                               
    print(f'encode 시간: {time.time() - t0:.1f}초')
    print('글자 수:', len(text))
    print('토큰 수:', len(ids))
    print(f'토큰 하나당 평균 글자 수: {len(text) / len(ids):.2f}')   
    print('복원 일치:', tok.decode(ids) == text)          
    sample = tok.encode('And God said unto Moses')
    print([tok.vocab[i] for i in sample])  