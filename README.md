# TFT_finance_sector

금융 하위섹터 주가지수의 TFT 기반 예측 (은행·보험·증권)
본 저장소는 한국 금융업 하위섹터(은행, 보험, 증권)의 주가지수를 대상으로 Temporal Fusion Transformer (TFT) 모델을 활용하여 구조적 변동 이후 구간의 시계열 예측, 변수 중요도 분석 및 해석 가능한 딥러닝 적용

1. 목적
구조적 변동점(Bai-Perron test 기반 2024-02-02) 이후 금융 하위섹터 주가지수의 시계열 패턴 예측

TFT 모델의 Attention, Variable Importance 등을 활용해 자기지수·외생변수·계절성이 예측 성능에 미치는 영향 정량화

전통 계량모형(VECM 등)과의 보완적 활용 가능성 검증

2. 코드 설명 및 차이점
파일명	주요 특징
TFT_subindex_of_finance_sector_240202	구조적 변동 이후 구간(2024-02-02 ~ 2024-12-30)을 집중 분석. Target: 보험지수
TFT_subindex_of_finance_sector_morevariables	외생변수(금리, 환율), 상위 지수(KOSPI, KOSDAQ) 추가 포함. 변수 확장 실험.
TFT_subindex_of_finance_sector_200301	전체 데이터의 30% 부분 테스트. 장기 구간에서 TFT 적용 실험.

3. 공통 분석 파이프라인
   데이터 전처리

금융 하위섹터 지수(은행, 보험, 증권) 및 외생변수(변동성, 국고채3/10년, 환율) 로드

구조적 변동 이후 구간 선택 or 전체 구간 사용

결측치 처리, 로그수익률, 변동성, 이동평균 등 파생 변수 생성

seasonality 변수(day_of_week, week_of_year, month 등) 추가

  TimeSeriesDataSet 생성

TimeSeriesDataSet 클래스 활용: target 설정(보험), known/unknown, categorical/real features 분리

max_encoder_length와 max_prediction_length 최적화

  TFT 모델 선언 & 학습

PyTorch Forecasting의 TemporalFusionTransformer 사용

학습률, hidden_size, attention_head_size, dropout, quantile loss 설정

Trainer로 학습, EarlyStopping 적용

  예측 및 해석

실제값 vs 예측값 시계열 plot

Variable Importance (encoder/decoder별)

Attention Weights (time index별 가중치)

5 day rolling window

예측 성능: MAE, MSE, p50/p90 error

4. 주요 결과 요약
예측 성능

구조적 변동 이후(3rd regime) 보험지수 target에서 TFT는 추세와 seasonality를 안정적으로 복원

급격한 충격 구간은 일부 오차 증가

변수 중요도

Encoder: 보험지수 자기값 > 이동평균 > 변동성 순

Decoder: days_from_start, day_of_week 등 계절성 변수 비중 높음

금리/환율 등 외생변수: 중요도 낮음, 모델 적응력에 따라 영향 차등 반영

Attention Mechanism

최근 데이터에 높은 가중치 집중, 충격구간 과거 데이터 선택적 강조

구조적 시사점

TFT는 구조적 변동 이후 구간의 계절성과 트렌드를 동시에 반영

5. 실행 환경
Python 3.8+

PyTorch Forecasting

pytorch-lightning

pandas, numpy, matplotlib

Colab 환경에서 실행

6. 참고

모델 구현 라이선스: MIT License (PyTorch Forecasting 기반)

분석 데이터: DataGuide MK2000 (상용 데이터)

참고 문헌: Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." International Journal of Forecasting 37.4 (2021): 1748-1764.
