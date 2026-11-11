# bible_LM

외부 구현 코드 없이 PyTorch 만으로 GPT 구조의 소형 언어모델을 직접 구현하고, 영어 KJV 성경 텍스트로 학습해 보는 프로젝트입니다.

## 역할 분담
이 프로젝트는 AI(Claude)와 협업하되, 코드를 이해하고 직접 조합해서 작성하는 것을 원칙으로 합니다.

**사용자 (이해하고 직접 작성)**
- Claude 의 줄별 설명을 읽고, 각 줄이 왜 필요한지 이해한 뒤 직접 조합해서 코드 파일에 작성
- 실행, 디버깅, 결과 확인
- 설계 방향과 진행 여부 결정

**Claude (학습 도우미 · 기록 담당)**
- 개념 설명: 텐서, 배치, Attention 등 원리를 예시와 함께 설명
- 코드 안내: 필요한 코드를 보여주되 각 줄에 무엇을 하고 왜 필요한지 주석으로 설명 (사용자가 읽고 이해한 뒤 직접 작성)
- 코드 리뷰: 작성한 코드를 읽고 오류 원인과 개선점 설명
- 기록: 시행착오(증상·원인·해결), 설계 결정, 환경 고정 사항을 `CLAUDE.md` 에, 날짜별 진행 기록을 이 README 에 정리

## 환경
- Windows, NVIDIA GeForce RTX 5070 (VRAM 12GB)
- conda 환경 `bible_lm`, Python 3.12
- PyTorch 2.11.0+cu128

## 구현 단계
- [x] 1. 데이터 로드, 글자 단위 토크나이저, train/val 분할, 배치 함수
- [x] 2. Bigram 모델 + 학습 루프 + 생성
- [x] 3. Self-Attention
- [x] 4. Multi-Head Attention + FeedForward
- [x] 5. Transformer 블록 쌓기 (Residual + LayerNorm)
- [x] 6. 학습 다듬기 (GPU, Dropout, 체크포인트, 파일 분리) — LR 스케줄은 아직
- [x] 7. (심화) BPE 토크나이저
- [ ] 8. (선택) 자동미분 엔진

## 파일 구조
| 파일 | 역할 |
|---|---|
| `model.py` | 모델 구조 설정값, 토크나이저 선택(`tokenizer_type`: 'char' / 'bpe'), 모델 클래스(Head, MultiHeadAttention, FeedForward, Block, LanguageModel) |
| `train.py` | 데이터 준비, 학습 루프, val 최저 시점 체크포인트 저장 |
| `generate.py` | 저장된 모델을 불러와 프롬프트 뒤로 문장 생성 |
| `bpe.py` | BPE 토크나이저 학습·encode·decode·저장 (`bpe.json`) |
| `attention_test.py` | Self-Attention 원리 실험 (평균 → 행렬 곱 → softmax, Q/K/V) |

## 결과 비교
loss 는 글자당 val 기준 (BPE 는 토큰당 loss ÷ 토큰당 글자 수 2.69 로 환산)

| 단계 | 모델 | 학습 | val (글자당) |
|---|---|---|---|
| 2 | Bigram | CPU, 10,000 스텝 | 약 2.26 (train, 단일 배치) |
| 3 | 임베딩 + Attention 헤드 1개 | CPU, 10,000 스텝 | 약 2.0 (train, 단일 배치) |
| 4 | 멀티헤드 4개 + FeedForward | CPU, 10,000 스텝 | 1.9027 |
| 5 | Transformer 블록 4층 | CPU, 10,000 스텝 | 1.7461 |
| 6 | 2.7M 파라미터, 글자 단위 | GPU, 5,000 스텝, 3.0분 | 1.3141 |
| 6 | 같은 모델 | GPU, 20,000 스텝, 15.2분 | 1.1898 |
| 7 | 2.9M 파라미터, BPE(vocab 574) | GPU, 5,000 스텝, 3.6분 | 1.0955 |
| 7 | 같은 모델 | GPU, 20,000 스텝, 15.2분 | **1.0606** (16,000 스텝 최저) |

생성 예시 (BPE 모델):
```
1:4 And Moses went up, and went out to battle, and said, What thy servant have gotten us? 1:5 And he must betray him, that thou mayest stand before us: for the LORD hath delivered us out of thine hand, so that the God of our fathers should be thine.
```

## 진행 기록

### 2026-11-10
**한 일**
- 개발 환경 구축: 프로젝트 전용 conda 환경 `bible_lm` 생성, PyTorch 2.11.0+cu128 설치, RTX 5070 인식 확인
- 1단계 완료
  - `bible.txt` 분석: 전체 4,330,136 글자, 고유 글자(vocab) 74개
  - 글자 단위 토크나이저 `encode` / `decode` 구현
  - 전체 텍스트를 텐서로 변환하고 train 90% / val 10% 로 분할
  - `get_batch` 구현 (block_size=8, batch_size=4): 입력 x 와 정답 y 가 한 글자씩 밀려 나오는 것 확인

**시행착오**
- base 환경에 PyTorch 가 없어서 `ModuleNotFoundError` → 프로젝트 전용 환경을 새로 만들어 해결
- RTX 50 시리즈는 CUDA 12.8 빌드(cu128) PyTorch 가 필요하고, Windows 에서는 `--index-url` 을 지정하지 않으면 CPU 전용 버전이 설치됨
- `pip install torch` 만 하면 numpy 가 없어 경고 발생 → numpy 별도 설치
- `set(file)` 로 고유 글자를 구하려다 0개가 나옴 → 파일 객체는 줄 단위로 순회하고 이미 `read()` 로 끝까지 읽은 상태였음. 읽어 둔 문자열 `set(content)` 로 해결
- VS Code 가 Microsoft Store Python 으로 실행됨 → `Python: Select Interpreter` 에서 `bible_lm` 선택으로 해결

### 2026-11-11
**한 일**
- 2단계: `BigramModel` + 학습 루프(AdamW) + `generate`. loss 4.73 → 약 2.26. 장:절 형식과 짧은 단어만 흉내
- 3단계: 과거 평균을 for 문 / 하삼각 행렬 곱 / softmax+마스크로 계산해 같은 결과 확인 → Q/K/V Self-Attention 헤드 → 토큰·위치 임베딩을 쓰는 `LanguageModel`. loss 약 2.0
- 4단계: loss 평가 함수(train/val 200 배치 평균), Multi-Head Attention(4개), FeedForward. val 1.90
- 5단계: Residual + LayerNorm 을 쓰는 `Block` 4층. val 1.75. David, Joab 등 인물명 등장
- 6단계: GPU 로 이동, 모델 확대(2.7M 파라미터, block 128, 6층×6헤드), Dropout, 체크포인트 저장. `model.py` / `train.py` / `generate.py` 로 분리해 학습 없이 생성 가능. 20,000 스텝 val 1.19
- 7단계: BPE 토크나이저 직접 구현. 병합 500번(vocab 574), 성경 전체 2.69배 압축, 복원 완전 일치. 같은 모델로 학습하니 3.6분 만에 글자 모델 15분 학습보다 좋은 결과
- BPE 장기 학습: 약 15,000 스텝부터 val 정체, train 만 계속 감소 → 과적합 시작 확인

**배운 점**
- 초기 loss 는 ln(vocab) 보다 약간 높은 게 정상 (임베딩이 랜덤으로 초기화되어 근거 없는 선호가 생김)
- Attention 은 순서를 모르기 때문에 위치 임베딩을 따로 더해야 함
- Dropout 은 같은 스텝 수에서는 오히려 val 을 약간 높일 수 있음 (학습을 일부러 어렵게 하므로). 시드 고정 없이 한 번 비교로는 결론 내기 어려움
- BPE 는 같은 block_size 로 더 긴 문맥을 보지만, 데이터가 토큰 수로 줄어들어 같은 스텝이면 더 많이 반복 학습 → 과적합이 더 빨리 옴
- 토크나이저가 다른 모델끼리는 토큰당 loss 를 그대로 비교할 수 없어 글자당으로 환산해야 함

**시행착오**
- `LanguageModel` 로 바꾼 뒤 `model.generate` 에서 `AttributeError` → `generate` 가 `BigramModel` 에만 있었음. `LanguageModel` 에 최근 block_size 글자만 잘라 넣는 `generate` 를 추가해 해결 (위치 임베딩이 block_size 개뿐이라 그보다 길면 안 됨)
