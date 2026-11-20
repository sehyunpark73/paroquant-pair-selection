# paroquant pair selection

## 배경
- [paroquant](https://arxiv.org/abs/2511.10645)는 group 안 channel을 pair로 묶어 rotation하고, 그 각도를 학습해 양자화 loss를 줄임
- 논문 motivation은 outlier channel과 normal channel을 묶는 것
- 허나 [현행 구현](https://github.com/z-lab/paroquant/blob/legacy/paroquant/optimize.py)은 pair를 random으로 고름
- 이에 저자는 [open review 답변](https://openreview.net/forum?id=1USeVjsKau)에서 independence를 지킨다면 pair 기준에 robust하다고 함
- 이 프로젝트는 pair를 고르는 기준만 바꿔 이 주장을 확인

## 요약
- [저자 표 재현](#저자-표-재현): 저자 환경이 재현되는지 → loss를 mse로 재면 재현됨
- [arm별 loss](#arm별-loss): pair를 고르는 기준에 robust한지 → 아님, rotation을 쌓으면 정렬 arm은 random보다 나쁨
- [연결성](#연결성): 차이가 연결성 때문인지 → 아님
- [pair 점수 유지](#pair-점수-유지): 처음 매긴 pair 점수가 rotation을 쌓은 뒤에도 남는지 → 낮아지지만, 이것만으로 차이가 설명되지 않음
- [oracle arm](#oracle-arm): range를 직접 보고 고르면 나은지 → rotation 1개에서 가장 좋지만, rotation을 쌓을수록 이득이 사라짐
- [rotation마다 다시 고르기](#rotation마다-다시-고르기): rotation마다 다시 고르면 이득이 남는지 → 현행 구현보다 좋지만, 총 step을 맞추면 현행 구현이 더 좋음

## 한계
- q_proj의 layer 10, 20, 4비트, group size 128 위주로 봄
- loss는 linear 하나의 출력이라, 모델 전체의 ppl은 보지 않음

## 남은 실험
- 모델 전체를 양자화해 ppl로도 arm 사이 차이가 나는지 확인
- 총 step을 맞춘 비교를 다른 모델과 seed에서도 확인

## 공통 설정
- 각 실험의 설정에 따로 적지 않으면 아래 값을 사용
#### 용어
- pair: 한 group 안 channel 둘, givens rotation으로 함께 돌림
- rotation: group 안 channel을 겹치지 않는 pair로 묶고, pair마다 각도 theta를 학습하는 한 단계
- dummy: 이미 쓴 pair를 건너뛰다 채우지 못해 rotation하지 않는 자리
- loss: linear 출력의 mse를 rtn의 mse로 나눈 %, 아래가 좋음
- arm: pair를 고르는 기준
- random: 섞어서 고름
- butterfly: fft의 butterfly 패턴으로 묶음, random도 정렬도 아닌 비교군
- magnitude_difference: 두 channel의 weight 크기 차이가 큰 순서, 논문 motivation
- magnitude_similarity: 두 channel의 weight 크기 차이가 작은 순서, motivation의 반대
- correlation: 두 channel의 weight 상관계수 절댓값이 큰 순서
- activation_magnitude_difference: weight 크기에 입력 크기를 곱한 값의 차이가 큰 순서
- 정렬 arm: random과 butterfly를 뺀, 점수 순서대로 고르는 arm
- 크기 차이 arm: magnitude_difference와 activation_magnitude_difference
- 현행 구현: random으로 고르고 모든 rotation의 theta를 한꺼번에 학습
#### 모델과 양자화
- qwen3-0.6b, llama-3.1-8b, mistral-7b-v0.3, gemma-2-9b의 layer 10, 20 q_proj
- 4비트 asymmetric min-max, group size 128
- calibration은 pile 1024 token 16개 (논문 128개)
#### 학습
- channel scaling alpha와 theta를 함께 학습
- adamw, learning rate 0.01, 200 step
- rotation 8개
- random은 seed 5개 평균, 나머지 arm은 seed 하나

## 실험

### 저자 표 재현
- 이 프로젝트의 구현이 저자 환경을 재현하는지 확인
- open review 답변 Q4 표의 scaling과 rotation 8개 loss를 다시 잼
#### 설정
- qwen3-0.6b layer 0의 q_proj, gate_proj
- scaling: rotation 없이 alpha만 학습
- l2: 논문 A.2 표기대로 출력 차이의 l2 norm 비율, mse 비율의 제곱근
#### 결과
![](figures/reproduction.png)
- scaling은 l2로 재면 저자 값보다 나쁘고, mse로 재면 가까움
- rotation 8개는 mse로 저자 값과 거의 같음
- 따라서 저자 표의 loss는 mse로 재현되고, 이후 loss는 모두 mse

### arm별 loss
- 저자 주장대로 pair를 고르는 기준과 상관없이 loss가 같은지 확인
- arm마다 rotation 수와 step 수를 바꿔 loss를 비교
#### 설정
- 첫 그림은 layer 0도 봄
- 두 번째 그림은 rotation 1, 2, 4, 8개
- 세 번째 그림은 llama-3.1-8b, mistral-7b-v0.3의 layer 10에서 800 step도 봄
- magnitude_similarity는 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로, 첫 그림 뒤로는 뺌
- oracle은 [oracle arm](#oracle-arm)에서 다룸
#### 결과
![](figures/arm_loss.png)
- rotation 8개에서 정렬 arm은 모든 layer에서 random보다 나쁨

![](figures/arm_loss_rotations.png)
- rotation 1개에서는 activation_magnitude_difference가 random보다 좋지만, rotation을 쌓을수록 크기 차이 arm은 random보다 나빠짐

![](figures/arm_loss_steps.png)
- 800 step으로 늘려도 크기 차이 arm과 random의 차이가 줄지 않음
- 따라서 pair를 고르는 기준에 따라 loss가 달라지고, rotation을 쌓으면 정렬 arm은 random보다 나쁨

### 연결성
- [arm별 loss](#arm별-loss)의 차이가 pair들이 channel을 얼마나 이어 주는지에서 오는지 확인
- 모든 rotation의 pair를 한 그래프로 겹치고, dummy는 빼고 연결성을 잼
#### 설정
- blocks: group당 connected component 수, 1이면 group 안 channel이 모두 이어짐
- lambda2: laplacian의 두 번째로 작은 고유값, 0이면 끊겨 있고 클수록 강하게 이어짐
- rotation 1, 2, 4, 8개, random도 seed 하나
#### 결과
![](figures/connectivity.png)
- rotation 1개에서는 연결성이 모두 같은데도, [arm별 loss](#arm별-loss) 두 번째 그림에서 loss가 arm마다 다름
- rotation 2개에서 correlation은 random만큼 이어지지만 loss는 가장 나쁨
- 따라서 연결성은 arm 사이 loss 차이를 설명하지 못함

### pair 점수 유지
- 정렬 arm은 원본 weight로 pair 점수를 한 번만 매기므로, 앞선 rotation이 weight를 바꾸면 점수가 맞지 않을 수 있음
- rotation마다 그 rotation의 pair 점수를 다시 잼
#### 설정
- llama-3.1-8b layer 10
- score_retention: 그 rotation 직전 weight로 잰 pair 점수를 첫 rotation 전 weight로 잰 점수로 나눈 비
#### 결과
![](figures/pair_score.png)
- 정렬 arm의 score_retention은 rotation을 쌓을수록 낮아짐
- 크기 차이 arm 둘은 score_retention이 다르지만 loss는 같음
- 따라서 처음 매긴 pair 점수는 rotation을 쌓을수록 낮아지지만, 이것만으로 arm 사이 차이가 설명되지 않음

### oracle arm
- 정렬 arm의 점수는 pair가 range를 실제로 얼마나 줄이는지와 다를 수 있음
- range를 직접 보는 oracle을 arm에 더함
#### 설정
- range: group마다 weight 최댓값과 최솟값의 차
- oracle: pair마다 각도 16개를 돌려 보고, range를 가장 많이 줄인 감소량이 큰 순서
#### 결과
- [arm별 loss](#arm별-loss) 두 번째 그림에서 oracle은 rotation 1개에서 가장 좋지만, rotation을 쌓을수록 random과의 차이가 사라짐
- [pair 점수 유지](#pair-점수-유지) 그림에서 oracle의 score_retention이 다른 정렬 arm보다 빨리 낮아짐
- 따라서 range를 직접 보고 고르면 이득이 있지만, 처음 매긴 점수가 낮아져 rotation을 쌓을수록 이득이 사라짐

### rotation마다 다시 고르기
- [oracle arm](#oracle-arm)에서 처음 매긴 점수가 낮아져 이득이 사라지므로, rotation마다 다시 고르면 이득이 남는지 확인
- 앞선 rotation을 건 weight로 pair를 다시 고르고, 앞선 theta를 고정한 채 그 rotation만 학습
#### 설정
- schedule: all은 처음 한 번 고르고 theta를 한꺼번에 학습, sequential_rescore는 rotation마다 다시 고르고 하나씩 학습
- all random이 현행 구현
- 첫 그림의 sequential_rescore는 rotation마다 200 step, 총 1600 step
- 두 번째 그림은 총 step을 맞추고, sequential_rescore는 총 step을 rotation마다 나눠 학습
- 두 번째 그림은 qwen3-0.6b, random도 seed 하나
- 두 번째 그림은 비공개 paro_pair 코드로 낸 결과라, 첫 그림과 값을 비교하지 않음
#### 결과
![](figures/rescore.png)
- sequential_rescore oracle은 모든 모델과 layer에서 현행 구현보다 좋음
- sequential_rescore random은 현행 구현보다 나쁨

![](figures/rescore_budget.png)
- 총 1600 step으로 맞추면 현행 구현이 sequential_rescore oracle보다 좋음
- 따라서 rotation마다 다시 고르면 현행 구현보다 좋지만, 총 step을 맞추면 현행 구현이 더 좋음
