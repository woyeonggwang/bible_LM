# bible_LM — CLAUDE.md

소형 언어모델(GPT 구조)을 외부 구현 코드 없이 직접 짜서 학습해 보는 프로젝트.

## 작업 규칙 (Claude)
- 사용자가 직접 구현한다. 모델 핵심 로직(모델 구조, 학습 루프 등)은 Claude 가 정답 코드를 먼저 주지 않고, 막힌 부분에 힌트부터 준다. 힌트로 안 풀리면 단계적으로 더 구체적으로 알려준다.
- 파이썬 문법·사용법 질문은 힌트 없이 바로 답한다. (2026-11-10 결정: 문법까지 힌트로 돌리면 결국 외부 AI 에 코드를 물어보게 되어 시간만 듦)
- 시행착오(증상·원인·해결), 확정된 설계 결정과 이유, 버전·환경 고정 사항, 미해결 과제는 이 파일에 기록한다.
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

## 데이터 정보
- `bible.txt` 전체 글자 수: 4,330,136 (2026-11-10 확인)
- `bible.txt` 고유 글자 수(vocab_size): 74 (2026-11-10 확인)
- train/val 분할: 앞 90% train, 뒤 10% val (섞지 않고 연속 구간으로 분할 — 문장이 깨지지 않게 하기 위함)

## 진행 상황
- 1단계 완료 (2026-11-10): `main.py` 에 데이터 로드, 글자 단위 encode/decode(stoi/itos), 텐서 변환, train/val 분할, `get_batch`(block_size=8, batch_size=4, y 는 x 를 한 칸 민 것) 구현. x/y 가 한 글자씩 밀려 나오는 것 확인
- 대문자 `X` 는 등장하지 않음. 특수문자로 `—`(em dash), `’`(둥근 아포스트로피) 포함

## 미해결 과제
- 학습 데이터: 프로젝트 폴더의 `bible.txt` 를 사용 중. 영어 KJV 판본인지, Gutenberg 라이선스 머리말·꼬리말 처리 여부 (확인 필요)
