# bible_LM

외부 구현 코드 없이 PyTorch 만으로 GPT 구조의 소형 언어모델을 직접 구현하고, 영어 KJV 성경 텍스트로 학습해 보는 프로젝트입니다.

## 환경
- Windows, NVIDIA GeForce RTX 5070 (VRAM 12GB)
- conda 환경 `bible_lm`, Python 3.12
- PyTorch 2.11.0+cu128

## 구현 단계
- [x] 1. 데이터 로드, 글자 단위 토크나이저, train/val 분할, 배치 함수
- [ ] 2. Bigram 모델 + 학습 루프 + 생성
- [ ] 3. Self-Attention
- [ ] 4. Multi-Head Attention + FeedForward
- [ ] 5. Transformer 블록 쌓기
- [ ] 6. 학습 다듬기 (Dropout, LR 스케줄, 체크포인트)
- [ ] 7. (심화) BPE 토크나이저
- [ ] 8. (선택) 자동미분 엔진

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
