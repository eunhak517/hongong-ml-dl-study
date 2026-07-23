<div align="center">

# 🤖 혼자 공부하는 머신러닝 + 딥러닝

### Machine Learning & Deep Learning Study Archive

머신러닝의 기초부터 딥러닝과 트랜스포머까지  
**직접 구현하고, 실행 결과를 해석하며 기록하는 학습 저장소입니다.**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Planned-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Planned-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

<br>

[![Last Commit](https://img.shields.io/github/last-commit/eunhak517/hongong-ml-dl-study?style=flat-square)](https://github.com/eunhak517/hongong-ml-dl-study/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/eunhak517/hongong-ml-dl-study?style=flat-square)](https://github.com/eunhak517/hongong-ml-dl-study)

<br>

<!-- PROGRESS_START -->
![Progress](https://img.shields.io/badge/Progress-1%20%2F%2028-2EA44F?style=for-the-badge)

```text
█░░░░░░░░░░░░░░░░░░░  4%
```

**완료 1개 · 전체 28개 절**
<!-- PROGRESS_END -->

<br>

**기준 교재:** 『혼자 공부하는 머신러닝+딥러닝』 개정판  
**작성자:** [eunhak517](https://github.com/eunhak517)

</div>

---

## 📌 저장소 소개

이 저장소는 『혼자 공부하는 머신러닝+딥러닝』을 학습하며 작성한  
**개인 실습 코드, 개념 정리, 결과 해석과 오류 해결 과정**을 기록합니다.

예제 코드를 실행하는 데서 끝내지 않고 다음 과정을 반복합니다.

1. 문제와 데이터 구조를 파악합니다.
2. 모델을 직접 구현하고 학습합니다.
3. 출력과 성능 지표의 의미를 해석합니다.
4. 전처리와 하이퍼파라미터가 결과에 미치는 영향을 확인합니다.
5. 과대적합·과소적합과 데이터 누수 가능성을 점검합니다.
6. 실패 원인과 해결 방법을 제 언어로 기록합니다.
7. 배운 내용을 로봇·피지컬 AI 프로젝트에 연결합니다.

> 이 저장소는 공식 예제 저장소가 아닌 **개인 학습 기록 저장소**입니다.  
> 교재 본문을 그대로 복제하지 않고, 직접 실행한 코드와 이해한 내용을 중심으로 정리합니다.


```text
✅ 03_02_linear_regression.ipynb
✅ 05_01_decision_tree.ipynb
❌ linear_regression.ipynb
❌ chapter3-practice.ipynb
```

---

## 🎯 학습 목표

- 지도 학습과 비지도 학습의 차이 이해
- 훈련·검증·테스트 데이터의 올바른 분리
- 데이터 전처리와 특성 스케일링
- 분류 및 회귀 문제 모델링
- 과대적합과 과소적합 진단
- 교차 검증과 하이퍼파라미터 탐색
- 트리 기반 앙상블 모델 활용
- 군집화와 차원 축소
- 인공 신경망과 심층 신경망 구현
- CNN을 활용한 이미지 분류
- RNN·LSTM·GRU를 활용한 순차 데이터 처리
- 트랜스포머와 대규모 언어 모델의 구조 이해
- 학습한 내용을 로봇 주행 데이터 분석에 적용

---

## 🗺️ 전체 학습 로드맵

<details open>
<summary><strong>Chapter 01 · 나의 첫 머신러닝</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 01-1 | 인공지능과 머신러닝, 딥러닝 |
| 01-2 | 코랩과 주피터 노트북 |
| 01-3 | 마켓과 머신러닝 |

**핵심 목표**

- 인공지능·머신러닝·딥러닝의 관계 이해
- Google Colab과 Jupyter Notebook 사용
- 특성, 샘플, 타깃의 의미 이해
- k-최근접 이웃을 활용한 첫 분류 모델 구현

</details>

<details open>
<summary><strong>Chapter 02 · 데이터 다루기</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 02-1 | 훈련 세트와 테스트 세트 |
| 02-2 | 데이터 전처리 |

**핵심 목표**

- 지도 학습과 비지도 학습 구분
- 샘플링 편향의 위험 이해
- NumPy 배열과 인덱싱 활용
- 훈련·테스트 세트 분리
- 층화 추출과 특성 표준화

</details>

<details open>
<summary><strong>Chapter 03 · 회귀 알고리즘과 모델 규제</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 03-1 | k-최근접 이웃 회귀 |
| 03-2 | 선형 회귀 |
| 03-3 | 특성 공학과 규제 |

**핵심 목표**

- 분류와 회귀의 차이 이해
- 결정계수와 회귀 오차 해석
- 과대적합과 과소적합 진단
- 선형·다항·다중 회귀 구현
- 릿지·라쏘 규제로 모델 복잡도 제어

</details>

<details>
<summary><strong>Chapter 04 · 다양한 분류 알고리즘</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 04-1 | 로지스틱 회귀 |
| 04-2 | 확률적 경사 하강법 |

**핵심 목표**

- 이진·다중 분류의 확률 예측
- 시그모이드와 소프트맥스 함수
- 손실 함수와 경사 하강법
- 점진적 학습과 에포크 조절

</details>

<details>
<summary><strong>Chapter 05 · 트리 알고리즘</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 05-1 | 결정 트리 |
| 05-2 | 교차 검증과 그리드 서치 |
| 05-3 | 트리의 앙상블 |

**핵심 목표**

- 결정 트리의 분할 기준과 해석
- 검증 세트와 교차 검증
- 그리드 서치와 랜덤 서치
- 랜덤 포레스트와 부스팅 모델

</details>

<details>
<summary><strong>Chapter 06 · 비지도 학습</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 06-1 | 군집 알고리즘 |
| 06-2 | k-평균 |
| 06-3 | 주성분 분석 |

**핵심 목표**

- 타깃이 없는 데이터 분석
- k-평균 군집화
- 적절한 군집 수 탐색
- PCA를 이용한 차원 축소
- 설명된 분산과 데이터 재구성

</details>

<details>
<summary><strong>Chapter 07 · 딥러닝을 시작합니다</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 07-1 | 인공 신경망 |
| 07-2 | 심층 신경망 |
| 07-3 | 신경망 모델 훈련 |

**핵심 목표**

- Fashion-MNIST 데이터셋 분류
- 뉴런, 층, 활성화 함수 이해
- 완전연결 신경망 구현
- 손실·검증 곡선 해석
- 드롭아웃, 모델 저장, 콜백 활용

</details>

<details>
<summary><strong>Chapter 08 · 이미지를 위한 인공 신경망</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 08-1 | 합성곱 신경망의 구성 요소 |
| 08-2 | 합성곱 신경망을 사용한 이미지 분류 |
| 08-3 | 합성곱 신경망의 시각화 |

**핵심 목표**

- 합성곱, 필터, 특성 맵
- 패딩, 스트라이드와 풀링
- CNN 기반 이미지 분류
- 가중치와 특성 맵 시각화

</details>

<details>
<summary><strong>Chapter 09 · 텍스트를 위한 인공 신경망</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 09-1 | 순차 데이터와 순환 신경망 |
| 09-2 | 순환 신경망으로 IMDB 리뷰 분류하기 |
| 09-3 | LSTM과 GRU 셀 |

**핵심 목표**

- 순차 데이터와 시퀀스 모델
- RNN의 은닉 상태
- 토큰화와 단어 임베딩
- LSTM·GRU 구조와 성능 비교

</details>

<details>
<summary><strong>Chapter 10 · 언어 모델을 위한 신경망</strong></summary>

<br>

| 절 | 학습 주제 |
|---|---|
| 10-1 | 어텐션 메커니즘과 트랜스포머 |
| 10-2 | 트랜스포머로 상품 설명 요약하기 |
| 10-3 | 대규모 언어 모델로 텍스트 생성하기 |

**핵심 목표**

- 셀프 어텐션과 멀티헤드 어텐션
- 위치 인코딩과 트랜스포머 블록
- Hugging Face 모델 활용
- 텍스트 요약과 생성
- 디코딩 전략 비교

> API 키와 비밀 정보는 노트북에 직접 작성하거나 GitHub에 커밋하지 않습니다.

</details>

---

## 📁 디렉터리 구조

```text
hongong-ml-dl-study/
├── README.md
├── requirements.txt
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── update-progress.yml
│
├── scripts/
│   └── update_progress.py
│
├── notebooks/
│   ├── 01_first_machine_learning/
│   ├── 02_data_handling/
│   ├── 03_regression_and_regularization/
│   ├── 04_classification/
│   ├── 05_tree_algorithms/
│   ├── 06_unsupervised_learning/
│   ├── 07_deep_learning/
│   ├── 08_convolutional_neural_networks/
│   ├── 09_recurrent_neural_networks/
│   └── 10_transformers_and_llm/
│
├── notes/
│   ├── concepts/
│   └── troubleshooting/
│
└── assets/
    ├── images/
    └── results/
```

---

## 📝 노트북 작성 원칙

각 노트북은 다음 구조를 기본으로 합니다.

```text
1. 학습 목표
2. 핵심 개념
3. 데이터 준비
4. 모델 구현
5. 성능 평가
6. 결과 시각화
7. 결과 해석
8. 오류와 해결 과정
9. 배운 점
```

기록할 내용:

- 입력 특성과 타깃
- 데이터 분리 방법
- 모델과 하이퍼파라미터
- 평가 지표와 의미
- 훈련 점수와 테스트 점수
- 과대적합 또는 과소적합 여부
- 오류 원인과 해결 과정
- 실습을 통해 이해한 핵심 내용

---

## 📚 전체 실습 완료 후 최종 정리

모든 절의 실습을 완료한 뒤, 전체 노트북을 다시 검토하여 다음 내용을 한 번에 정리할 예정입니다.

| 정리 항목 | 내용 |
|---|---|
| 절별 학습 내용 | 각 절에서 배운 핵심 개념 |
| 사용 모델 | 분류·회귀·군집·신경망 모델 |
| 주요 실험 결과 | 정확도, 결정계수, 손실 등 핵심 수치 |
| 모델 비교 | 전처리·하이퍼파라미터·알고리즘별 차이 |
| 오류와 해결 | 반복적으로 발생한 오류와 해결법 |
| 최종 회고 | 가장 중요했던 개념과 성장한 부분 |
| 프로젝트 적용 | 로봇 주행 불안정성 평가에 적용할 내용 |

> 진행 중에는 각 노트북 안에 결과를 기록하고,  
> 모든 실습이 끝난 뒤 README에 전체 결과를 요약합니다.

---

## 🔗 프로젝트 연계

학습한 머신러닝·딥러닝 개념은 다음 대표 프로젝트에 적용할 예정입니다.

### [`robot-driving-instability-analysis`](https://github.com/eunhak517/robot-driving-instability-analysis)

- IMU·Odometry·Camera 데이터 전처리
- 센서 타임스탬프 동기화
- 시계열 윈도우 생성
- 주행 불안정성 특징 추출
- 안정·불안정 라벨 기준 설계
- 분류·회귀 베이스라인 구축
- 데이터 누수 방지와 홀드아웃 평가
- 모델 결과와 물리적 의미 연결

혼공머신·딥러닝 학습을 완료한 뒤, 이 저장소에서 익힌 모델 평가와 실험 설계 방법을 프로젝트에 적용합니다.

---

## 💻 실행 방법

### 저장소 복제

```bash
git clone https://github.com/eunhak517/hongong-ml-dl-study.git
cd hongong-ml-dl-study
```

### 가상환경 생성

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 패키지 설치

```bash
pip install -r requirements.txt
```

### Jupyter 실행

```bash
jupyter lab
```

---

## 🔐 보안 원칙

다음 정보는 GitHub에 업로드하지 않습니다.

- API 키와 액세스 토큰
- `.env` 파일
- 학교 계정과 개인정보
- 공개가 금지된 데이터
- 대용량 원본 데이터셋
- 개인정보가 표시된 수료증 원본

---

## 🧾 커밋 메시지 규칙

| Prefix | 용도 | 예시 |
|---|---|---|
| `study` | 새로운 학습 내용 | `study: add KNN regression notebook` |
| `docs` | README·설명 수정 | `docs: update chapter notes` |
| `fix` | 코드 오류 수정 | `fix: correct matplotlib show call` |
| `refactor` | 코드 구조 개선 | `refactor: organize preprocessing cells` |
| `chore` | 환경·폴더 설정 | `chore: add Python gitignore` |

---

## 📚 참고 자료

- 『혼자 공부하는 머신러닝+딥러닝』 개정판
- 저자 공식 예제 코드 저장소
- scikit-learn 공식 문서
- NumPy 공식 문서
- Matplotlib 공식 문서
- TensorFlow 공식 문서
- PyTorch 공식 문서
- Hugging Face Transformers 문서

---

## ⚖️ 저작권 및 이용 안내

이 저장소는 개인 학습과 비상업적 포트폴리오 기록을 목적으로 합니다.

- 교재의 저작권은 저자와 출판사에 있습니다.
- 교재 본문, 삽화, 표와 확인 문제를 그대로 복제하지 않습니다.
- 공개된 실습 코드는 출처를 표시하고 직접 작성한 설명과 구분합니다.
- 교재를 대체할 수 있을 정도의 내용을 재배포하지 않습니다.

---

<div align="center">

### 🚀 꾸준히 배우고, 직접 실험하고, 결과로 증명하기

**Machine Learning → Deep Learning → Physical AI**

</div>
