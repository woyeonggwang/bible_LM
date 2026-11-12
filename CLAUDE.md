# bible_LM — CLAUDE.md

소형 언어모델(GPT 구조)을 외부 구현 코드 없이 직접 짜서 학습해 보는 프로젝트.

## 작업 규칙 (Claude)
- 코드 안내 방식 (2026-11-11 변경): Claude 가 필요한 코드 전체를 한 코드 블록에 모아 보여주고, 각 줄마다 "이 줄이 무엇을 하고 왜 필요한지"를 주석으로 붙인다. 사용자는 주석을 읽고 이해한 뒤 직접 main.py 에 작성한다.
  - 처음에는 줄마다 코드 블록을 나누고 설명을 본문에 썼으나 가독성이 떨어져 주석 방식으로 바꿈 (2026-11-11)
  - 이유: 힌트만으로는 결국 외부 AI 에 코드를 물어보게 되어 시간만 들고, 통째로 받으면 이해 없이 붙여넣게 됨. 줄별 설명 + 직접 조합이 이해와 속도 둘 다 챙기는 방식
  - (이전 방식: 2026-11-10 에는 핵심 로직은 힌트 먼저, 문법은 바로 답하기로 했었음)
- 사용자가 작성한 코드는 Claude 가 main.py 를 직접 읽어 확인·리뷰한다.
- 시행착오(증상·원인·해결), 확정된 설계 결정과 이유, 버전·환경 고정 사항, 미해결 과제는 이 파일에 기록한다.
  - 시행착오에는 개념·환경·설계에서 배울 점이 있는 것만 적는다. 오타, 줄을 잘못된 위치에 쓴 것 같은 단순 작성 실수는 CLAUDE.md·README 어디에도 기록하지 않는다 (2026-11-11, 사용자 요청)
- 확인된 것만 적는다. 확인되지 않은 것은 "(확인 필요)"로 표시한다.
- README.md 에는 날짜별 진행 기록(한 일, 겪은 시행착오 요약)을 추가한다.
- 백업 파일은 사용자가 요청할 때만 만든다.

## 확정된 설계 결정
- 프레임워크는 PyTorch 만 사용하고, 모델·토크나이저·학습 루프는 직접 구현한다. (이유: 구조를 직접 이해하기 위해 외부 구현을 받아 쓰지 않기로 함)
- 개발 도구: VS Code. 초반 실험은 노트북(.ipynb), 동작 확인 후 .py 로 정리.

## 구현 단계 (계획)
1. 데이터 로드 + 글자 단위 토크나이저(encode/decode), train/val 분리, 배치 함수
2. Bigram 모델 + 학습 루프 + 생성 함수
3. Self-Attention (Q/K/V, causal mask)
4. Multi-Head Attention + FeedForward
5. Transformer 블록 (Residual, LayerNorm, 위치 임베딩), 층 쌓기
6. 학습 다듬기 (Dropout, LR 스케줄, 검증 loss, 체크포인트)
7. (심화) BPE 토크나이저 직접 구현
8. (선택) 자동미분 엔진 직접 구현

## 버전·환경 고정 사항
- 머신: 데스크톱 Windows, RTX 5070 (VRAM 12GB)
- conda 환경 이름: `bible_lm` (base 환경은 사용하지 않음)
- Python: 3.12 (`conda create -n bible_lm python=3.12` 로 생성)
- PyTorch: 2.11.0+cu128 (CUDA 12.8 빌드), `torch.cuda.is_available()` = True, GPU 인식 확인 (2026-11-10)
- PyTorch 설치 명령: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  - RTX 50 시리즈(Blackwell, sm_120)는 cu128 이상 빌드가 필요함
  - Windows 에서는 `--index-url` 없이 설치하면 PyPI 의 CPU 전용 torch 가 설치됨
- torchvision / torchaudio 는 설치하지 않음 (텍스트만 다루므로 불필요)

## 시행착오 기록
### 2026-11-10 base 환경에 torch 없음
- 증상: `(base)` 에서 `import torch` 시 `ModuleNotFoundError: No module named 'torch'`
- 원인: base 환경(Python 3.13.9)에 PyTorch 미설치
- 해결: 프로젝트 전용 conda 환경 `bible_lm`(Python 3.12)을 만들고 cu128 빌드 PyTorch 설치

### 2026-11-10 NumPy 경고
- 증상: `import torch` 시 `UserWarning: Failed to initialize NumPy: No module named 'numpy'`
- 원인: `pip install torch` 만으로는 numpy 가 함께 설치되지 않음
- 해결: `pip install numpy` (설치 후 경고가 사라지는지 확인 필요)

### 2026-11-10 고유 글자 수가 0으로 나옴
- 증상: `chars = sorted(set(file))` 로 고유 글자를 구했더니 `vocab_size` 가 0
- 원인: `set(file)` 은 파일 객체를 줄 단위로 순회함(글자 단위 아님). 게다가 바로 앞의 `file.read()` 가 파일 끝까지 읽어서 더 읽을 내용이 없어 빈 set 이 됨
- 해결: 이미 읽어 둔 문자열로 `set(content)` 를 사용

### 2026-11-10 VS Code 가 bible_lm 이 아닌 다른 Python 으로 실행됨
- 증상: VS Code 실행 버튼으로 돌리니 `C:\Users\USER\AppData\Local\Microsoft\WindowsApps\python3.12.exe` (Microsoft Store Python)로 실행됨. 터미널 프롬프트도 `(base)`
- 원인: VS Code 의 Python 인터프리터가 `bible_lm` 환경으로 선택되어 있지 않음
- 해결: Ctrl+Shift+P → `Python: Select Interpreter` → `bible_lm` 선택. 이후 `C:\Users\USER\anaconda3\envs\bible_lm\python.exe` 로 실행되는 것 확인 (2026-11-10)

### 2026-11-11 LanguageModel 에 generate 가 없다는 AttributeError
- 증상: 학습 루프는 끝까지 돈 뒤 `model.generate(start, max_new_tokens=300)` 에서 `AttributeError: 'LanguageModel' object has no attribute 'generate'`
- 원인: `generate` 는 `BigramModel` 안에만 정의되어 있고, 3단계에서 새로 만든 `LanguageModel` 에는 `__init__`/`forward` 만 있음. 사용하는 모델을 바꾸면서 메서드가 따라오지 않음 (traceback 이 torch 의 module.py 를 가리키는 건 nn.Module 의 `__getattr__` 가 없는 속성을 찾다 에러를 내기 때문)
- 해결: `LanguageModel` 에 `generate` 추가. Bigram 것을 그대로 복사하지 않고 `idx[:, -block_size:]` 로 최근 block_size 글자만 잘라 forward 에 넣음 — 위치 임베딩이 block_size 개 자리뿐이고 `tril` 도 block_size×block_size 라서 길이가 넘으면 IndexError 가 날 것 (코드상 판단). 적용 후 300 글자 생성이 정상 동작하는 것 확인 (2026-11-11)

## 데이터 정보
- `bible.txt` 전체 글자 수: 4,330,136 (2026-11-10 확인)
- `bible.txt` 고유 글자 수(vocab_size): 74 (2026-11-10 확인)
- 대문자 `X` 는 등장하지 않음. 특수문자로 `—`(em dash), `’`(둥근 아포스트로피) 포함
- train/val 분할: 앞 90% train, 뒤 10% val (섞지 않고 연속 구간으로 분할 — 문장이 깨지지 않게 하기 위함)

## 진행 상황
- 1단계 완료 (2026-11-10): `main.py` 에 데이터 로드, 글자 단위 encode/decode(stoi/itos), 텐서 변환, train/val 분할, `get_batch`(block_size=8, batch_size=4, y 는 x 를 한 칸 민 것) 구현. x/y 가 한 글자씩 밀려 나오는 것 확인
- README.md 진행 기록 작성 시작 (2026-11-10)
- 2단계 완료 (2026-11-11): `BigramModel`(nn.Embedding(vocab_size, vocab_size)), `forward(idx, targets=None)`, 학습 루프(AdamW lr=1e-3, batch_size=32, 10,000 스텝), `generate`(softmax + multinomial) 구현
  - 학습 전 loss 4.73 → 10,000 스텝 후 약 2.26 (단일 배치 기준이라 스텝마다 흔들림 있음)
  - 생성 결과: `134: s 3 Bey win Th hJome ...` 처럼 장:절 번호 형식, the/and/of 같은 짧은 단어는 흉내내지만 의미 없는 문장. 직전 글자 하나만 보는 Bigram 의 한계
- 3단계 완료 (2026-11-11)
  - 3-1: `attention_test.py` 에서 과거 평균을 for 문 / 하삼각 행렬 곱 / softmax+마스크 세 방식으로 계산해 결과 동일 확인
  - 3-2: `attention_test.py` 에서 Q/K/V(nn.Linear, head_size=16) 로 Self-Attention 헤드 하나 계산, 가중치가 균등하지 않고 미래 위치는 0 인 것 확인
  - 3-3: `main.py` 에 `Head` 클래스, `LanguageModel`(token + position embedding, n_embd=32, 헤드 1개, lm_head) 추가. generate 에서 `idx[:, -block_size:]` 로 잘라 사용
  - 결과: loss 약 2.26(Bigram) → 약 2.0. 생성문에 And, to the, of 등 실제 단어 증가
- 4단계 완료 (2026-11-11): `estimate_loss`(train/val 각 200 배치 평균, @torch.no_grad), `MultiHeadAttention`(4 헤드 × 8), `FeedForward`(32→128→ReLU→32) 추가
  - 결과 (10,000 스텝): train 1.7764, val 1.9027
  - 생성문: `15:44 The ceoe unto the lise thy fest, and give the kin.` 처럼 unto/thy 등 성경 어휘와 장:절 형식, 절마다 줄바꿈
- 5단계 완료 (2026-11-11): `MultiHeadAttention` 에 proj(nn.Linear) 추가, `Block`(Pre-LayerNorm + Residual: x = x + sa(ln1(x)), x = x + ffwd(ln2(x))), n_head=4, n_layer=4, 최종 `ln_f` 추가
  - 결과 (10,000 스텝): train 1.6123, val 1.7461
  - 생성문: Joab, David, king of Is(rael), And they, I will 등 성경 인물명·구절 등장
- 6단계 1차 완료 (2026-11-11): GPU(cuda) 이동, Dropout(0.2: Head 가중치, MHA proj 뒤, FeedForward 끝), 모델 확대, 학습 후 `model.pt` 저장
  - 설정: batch_size=64, block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.2, lr=3e-4, 5,000 스텝
  - 파라미터 2.719178M, model.pt 약 13MB
  - 결과: train 1.0342, val 1.2824 (5단계 val 1.7461 대비 크게 개선). train-val 차이 약 0.25 로 5단계(약 0.13)보다 벌어짐
  - 생성문: `The word of the LORD is the LORD.` 처럼 문법에 맞는 구절, thou/shewed/saith 등 KJV 어휘
  - 이 1차 학습은 MultiHeadAttention 의 proj 뒤 dropout 이 코드상 빠진 상태(`return self.proj(out)`)였음
- 6단계 마무리 (2026-11-11): 파일 분리
  - `model.py`: 모델 구조 설정값(device, block_size, n_embd, n_head, n_layer, dropout), 토크나이저, 모델 클래스. 구조 값은 학습·생성이 반드시 같아야 model.pt 를 불러올 수 있어서 여기에 둠
  - `train.py`: 데이터 준비, 학습 설정값(batch_size, max_steps, eval_interval, lr), 학습 루프, model.pt 저장 (구 main.py)
  - `generate.py`: model.pt 를 load_state_dict 로 불러와 eval() 후 프롬프트 뒤로 생성. 학습 없이 바로 생성되는 것 확인
  - BigramModel 코드는 삭제 (git 기록으로 보존)
  - 분리 후 재학습 (MHA dropout 적용): train 1.0746, val 1.3141, 학습 시간 3.0분 (time.time() 실측, 생성 시간 포함)
    - 1차(MHA dropout 누락) val 1.2824 보다 약간 높음. dropout 이 더 많이 걸려 5,000 스텝 안에서는 수렴이 느린 것으로 보이나, 시드를 고정하지 않아 실행마다 편차가 있으므로 한 번의 비교로 결론 내기 어려움 (확인 필요)
- 장기 학습 실험 (2026-11-11): 같은 설정으로 max_steps=20,000
  - 결과: train 0.9041, val 1.1898 (20,000 스텝에서 최저), 학습 15.2분. model.pt 는 이 20,000 스텝 버전으로 저장됨
  - val 감소폭이 줄어듦: 5,000→10,000 스텝 약 -0.07, 10,000→15,000 약 -0.02, 15,000→20,000 약 -0.03. 아직 val 이 올라가는 과적합 구간은 없음
  - train-val 차이 0.24(5,000) → 0.29(20,000)로 천천히 벌어지는 중
  - 생성문: `22:7 Then said the disciples ...`, `33:30 Behold, I will bring the king of`, Selah 등 문장 구조가 자연스러워짐
  - 이 결과를 글자 단위 모델의 기준점으로 삼아 7단계 BPE 와 비교 (BPE 는 토큰당 loss 라 글자당으로 환산해야 비교 가능)
- 7-1단계 완료 (2026-11-11): `bpe.py` 에 BPE 학습(`merge`, `train_bpe`) 구현
  - 설계: 정규식 `r" ?\w+| ?[^\w\s]+|\s+"` 로 단어 조각(앞 공백 포함)으로 나눈 뒤 조각별 등장 횟수로 쌍을 셈 → 단어 경계를 넘는 병합은 없음. 400만 글자를 매번 훑지 않아도 돼서 빠름
  - 기본 토큰은 기존과 같은 글자 74개, 병합 500번 → vocab 574
  - 첫 병합: 'th'(153,434번) → ' th' → ' the' → ' a' → 'nd' ... 22번째에 '\n\n'(절 사이 빈 줄)
  - 가장 긴 토큰: ' Jerusalem', ' therefore', ' according', ' children', ' Israel' 등
- 7-2단계 완료 (2026-11-11): `bpe.py` 에 `BPETokenizer`(encode_word + 단어 캐시, encode, decode, save/load JSON) 구현, `bpe.json` 저장
  - encode 는 단어 조각마다 '가장 먼저 학습된 병합 규칙(번호가 가장 작은 것)'부터 적용. JSON 은 튜플 키를 못 써서 merges 를 [a, b, 새번호] 리스트로 저장
  - 성경 전체: 4,330,136 글자 → 1,607,963 토큰, 토큰당 평균 2.69 글자, encode 0.4초, decode 후 원문과 완전히 일치 (True)
  - 'And God said unto Moses' → ['A', 'nd', ' God', ' said', ' unto', ' Moses']. 문장 맨 앞 'And' 는 앞 공백이 없어서 ' And' 토큰과 다르게 쪼개짐
- 7-3단계 완료 (2026-11-11): `model.py` 에 `tokenizer_type`('char'/'bpe') 전환 추가, BPE 모델은 `model_bpe.pt` 로 따로 저장(글자 모델 model.pt 보존). train.py 에서 토큰당 loss ÷ 토큰당 글자 수(2.69)로 글자당 val 출력
  - 설정은 6단계와 동일(block 128 토큰 ≈ 344 글자, embd 192, 6층×6헤드, dropout 0.2), vocab 574 로 파라미터 2.911678M
  - 결과 (5,000 스텝, 3.6분): train 2.2141, val 2.9501 (토큰당), val(글자당) 1.0955
  - 글자 모델 5,000 스텝 val 1.3141, 20,000 스텝(15.2분) val 1.1898 보다 낮음 → 더 짧은 학습으로 더 좋은 결과
  - 5,000 스텝에서 val 은 아직 감소 중 (과적합으로 다시 오르는 구간 없음)
  - 생성문: `Then shall ye make me a burnt offering for a young lion`, `Thus sware unto Ahab` 등 긴 구절이 이어짐. 'whichhe', 'bifmes' 같은 조각 조합 오류는 남아 있음
  - 주의: train/val 분할을 토큰 위치 기준 90% 로 하므로 글자 모델과 val 구간 경계가 완전히 같지는 않음 (비교 시 참고)
- BPE 장기 학습 실험 (2026-11-11): max_steps=20,000, 15.2분, val 최저 시점 저장 로직 추가
  - val 최저: 16,000 스텝 val 2.8561 (글자당 1.0606). 14,500 스텝 이후 val 은 2.856~2.872 사이에서 정체·소폭 상승, train 은 계속 감소 (1.8556 → 1.7705) → 과적합 시작 구간
  - 최종 20,000 스텝: train 1.7705, val 2.8689 (글자당 1.0653). train-val 차이 토큰당 약 1.10 (5,000 스텝 때 약 0.74)
  - 글자 모델은 20,000 스텝까지 val 이 계속 감소했던 것과 달리, BPE 는 약 15,000 스텝에서 한계 → 같은 스텝이면 데이터를 약 2.7배 더 반복해서 보기 때문
  - 이번 실행의 model_bpe.pt 는 최저 시점(16,000)이 아니라 마지막(20,000) 스텝 모델로 저장됨 (best_val 초기화 위치 문제로 매 평가마다 저장됨). 글자당 val 차이는 약 0.005
  - 생성문: `And Moses went up, and went out to battle`, `for the LORD hath delivered us out of thine hand`, `But David and his men came to Ziph` 등
- LR 스케줄 실험 (2026-11-12): train.py 에 warmup(500 스텝, 0→lr 직선) + cosine decay(lr 3e-4 → min_lr 3e-5) 추가, best_val 초기화를 반복문 밖으로 옮김(이후 val 이 낮아질 때만 저장되는 것 확인)
  - BPE, max_steps=15,000, 9.3분: 최저 val 2.8645 @15,000 (글자당 1.0637), train 1.9853
  - 고정 lr 3e-4 의 같은 15,000 스텝 val 2.8578 (글자당 1.0612) 보다 약간 높음 → val 개선 효과는 확인되지 않음 (차이 0.007, 시드 미고정이라 실행 편차 범위일 수 있음)
  - train-val 차이는 고정 lr 15,000 스텝 약 1.00 → 이번 약 0.88 로 작음 (과적합 덜 함). 후반부 val 이 정체 없이 끝까지 조금씩 감소
  - 원인 추정: 같은 최대 lr 에서 cosine 으로 줄이면 평균 학습률이 낮아져 학습이 덜 진행됨 (확인 필요). 스케줄을 쓰면 최대 lr 을 더 높게 잡는 것이 일반적
- LR 스케줄 실험 2 (2026-11-12): 최대 lr 1e-3, min_lr 1e-4, warmup 500 + cosine, 15,000 스텝, 9.1분
  - 최저 val 2.8442 @14,500 (글자당 1.0562). 기존 최고(고정 lr 3e-4, 2.8561 @16,000)보다 0.012 낮음 — 차이가 작아 실행 편차일 수 있음
  - 7,500 스텝에서 이미 val 2.8472 로 기존 최고 수준 도달 → 같은 성능에 도달하는 스텝 수가 약 절반
  - 7,500 스텝 이후 val 은 2.844~2.860 에서 정체, train 은 1.81 → 1.62 로 계속 감소. 최종 train-val 차이 약 1.23 으로 가장 큼
  - 해석: 학습 속도는 lr 로 빨라졌지만 최저 val 은 거의 그대로 → 현재 병목은 최적화가 아니라 데이터 양(약 145만 학습 토큰 vs 2.9M 파라미터)에 따른 과적합
  - 생성문: `And they departed from Bashan before the LORD.`, 인명 나열(Salai, Zechariah, Hanani, Meshelemiah) 등. 'Pharistines'(Pharisees+Philistines 혼합) 같은 오류
- 8단계 완료 (2026-11-12): 자동미분 엔진 직접 구현
  - `autograd.py`: `Value` 클래스(+, *, **, tanh 와 각 연산의 _backward, 위상 정렬 기반 backward). 같은 값이 여러 곳에 쓰이면 기울기를 += 로 합산
  - 검증: L = tanh(a*b + c)*a + b**2/c (a=2, b=-3, c=7) 의 L 과 a/b/c 기울기가 PyTorch(double) 와 소수점 16자리까지 일치
  - `mlp.py`: Value 만으로 Neuron/Layer/MLP(3→4→4→1, 파라미터 41개) 구현, MSE + SGD(lr 0.05) 100 스텝 학습. loss 0.4883(10 스텝) → 0.0111, 예측 [0.965, -0.937, -0.956, 0.937] (정답 [1, -1, -1, 1])
- 프로젝트 마무리 (2026-11-12): 1~8단계 완료. 최고 성능은 BPE + warmup·cosine(lr 1e-3) 글자당 val 1.0562. 현재 병목은 데이터 양(과적합)
- 다음: 소형 LLM 파인튜닝 (별도 프로젝트로 진행할지, 폴더 위치 등은 확인 필요)

## 결과 비교 (2~5단계는 같은 설정: block_size=8, batch_size=32, lr=1e-3, 10,000 스텝, CPU)
| 단계 | 모델 | loss |
|---|---|---|
| 2 | Bigram | 약 2.26 (단일 배치) |
| 3 | 임베딩 32 + 헤드 1개 | 약 2.0 (단일 배치) |
| 4 | 멀티헤드 4개 + FeedForward | train 1.7764 / val 1.9027 (200 배치 평균) |
| 5 | 블록 4층 (Residual + LayerNorm) | train 1.6123 / val 1.7461 |
| 6 | GPU, 2.7M 파라미터 (block 128, embd 192, 6층×6헤드, dropout 0.2, 5,000 스텝), MHA dropout 누락 | train 1.0342 / val 1.2824 |
| 6 | 위와 같음, MHA dropout 적용, 학습 3.0분 | train 1.0746 / val 1.3141 |
| 6 | 위와 같음, 20,000 스텝, 학습 15.2분 | train 0.9041 / val 1.1898 |
| 7 | BPE(vocab 574), 2.9M 파라미터, 5,000 스텝, 학습 3.6분 | 토큰당 train 2.2141 / val 2.9501, **글자당 val 1.0955** |
| 7 | BPE, 20,000 스텝, 학습 15.2분 | 최저 val 2.8561 @16,000 스텝, **글자당 1.0606** (최종 20,000: 글자당 1.0653) |
| 7 | BPE + warmup·cosine (lr 3e-4→3e-5), 15,000 스텝, 9.3분 | 최저 val 2.8645 @15,000, 글자당 1.0637 |
| 7 | BPE + warmup·cosine (lr 1e-3→1e-4), 15,000 스텝, 9.1분 | 최저 val 2.8442 @14,500, **글자당 1.0562** |


## 확인된 사실
- 학습 전 Bigram 의 초기 loss 는 ln(74) ≈ 4.30 이 아니라 약 4.7~4.8 이 정상: 4.30 은 모든 logit 이 같을 때의 값이고, nn.Embedding 은 N(0,1) 랜덤으로 초기화되어 근거 없는 선호가 생기므로 기대 loss ≈ ln(V) + 0.5 (2026-11-11, 실측 4.73)

## 미해결 과제
- 학습 데이터: 프로젝트 폴더의 `bible.txt` 를 사용 중. 영어 KJV 판본인지, Gutenberg 라이선스 머리말·꼬리말 처리 여부 (확인 필요)
