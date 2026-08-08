# HE-ML Workstation Training

이 도구는 `ddoloh/HE-ML` 저장소의 기존 NumPy LSTM과 Multi-Scheme HE 모듈을 그대로 사용하면서,
워크스테이션에서 **실제 공개 시계열 데이터 10종을 다운로드 → 로컬 저장 → 학습 → 테스트 → 추론 → HE 입력 round-trip 검증**할 수 있게 합니다.

## 포함 데이터셋

1. `ett_m1` — ETTm1, 15분 전력 변압기 데이터, target=`OT`
2. `ett_m2` — ETTm2, 15분 전력 변압기 데이터, target=`OT`
3. `h2o` — 호주 보건 시스템 월별 지출
4. `fuel_consumption` — 스페인 월별 연료 소비
5. `air_quality` — Valencia 시간별 대기질
6. `website_visits` — 일별 웹사이트 방문
7. `bike_sharing` — Washington DC 시간별 자전거 대여
8. `australia_tourism` — 호주 분기별 관광 수요
9. `uk_daily_flights` — 영국 일별 항공편 수
10. `vic_electricity` — Victoria 반시간별 전력 수요

데이터 파일은 `data/real_datasets/`에 저장됩니다.
학습 모델은 `models/workstation_lstm/`, 결과는 `results/workstation/`에 저장됩니다.

## 설치

HE-ML 저장소 루트에서:

```bash
python3 -m venv .venv-workstation
source .venv-workstation/bin/activate
pip install -r requirements.txt
pip install -r requirements-workstation.txt
```

## 데이터 다운로드

전체 10종:

```bash
python workstation_train.py download --dataset all
```

하나만:

```bash
python workstation_train.py download --dataset ett_m1
```

강제 재다운로드:

```bash
python workstation_train.py download --dataset all --force
```

## 학습

빠른 테스트:

```bash
python workstation_train.py train \
  --dataset ett_m1 \
  --epochs 3 \
  --max-points 1000 \
  --seq-len 24
```

조금 더 본격적인 학습:

```bash
python workstation_train.py train \
  --dataset ett_m1 \
  --epochs 15 \
  --max-points 5000 \
  --seq-len 96 \
  --lr 0.01
```

학습은 시간순으로 80% train / 20% test로 나눕니다.
결과는 `results/workstation/<dataset>.json`에 저장됩니다.

## 추론

학습 후:

```bash
python workstation_train.py infer \
  --dataset ett_m1 \
  --scheme ckks \
  --seq-len 96
```

HE scheme은 다음을 지원합니다.

```text
ckks
bfv
tfhe
paillier
```

예:

```bash
python workstation_train.py infer --dataset ett_m1 --scheme bfv
python workstation_train.py infer --dataset ett_m1 --scheme tfhe
python workstation_train.py infer --dataset ett_m1 --scheme paillier
```

## 10개 전체 빠른 benchmark

```bash
python workstation_train.py benchmark \
  --datasets all \
  --epochs 3 \
  --max-points 1500 \
  --seq-len 24
```

결과:

```text
results/workstation/benchmark.json
```

## 중요한 HE 범위

현재 `ddoloh/HE-ML`의 `multi_scheme_he` 구현은 CKKS/BFV/TFHE/Paillier의 인터페이스와 암복호화 경로를 제공하지만,
현재 LSTM adapter 자체는 암호문 상태에서 LSTM의 모든 행렬 연산을 실제 FHE로 수행하는 구현이 아닙니다.

따라서 `infer` 명령은:

1. 마지막 입력 window 생성
2. HE encrypt
3. HE decrypt
4. 복원된 입력을 기존 LSTM에 넣어 다시 추론
5. 원본 입력과 복호화 입력의 오차 및 예측값 차이 측정

을 수행합니다.

즉 **HE round-trip / 입력 보존 검증**이지 "암호문 위에서 LSTM 전체를 수행하는 완전한 FHE inference"는 아닙니다.

완전한 FHE LSTM inference가 목적이라면 다음 단계에서 LSTM의 gate 연산을 각 scheme의 실제 homomorphic primitive로 변환해야 합니다.
