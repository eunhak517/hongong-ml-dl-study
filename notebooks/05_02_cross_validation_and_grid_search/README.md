# 05-2. 교차 검증과 그리드 서치

와인 분류 결정 트리의 성능을 검증 세트와 교차 검증으로 평가하고, `GridSearchCV`와 `RandomizedSearchCV`로 하이퍼파라미터를 선택한 실습입니다.

- 실습 노트북: [`05_02_cross_validation_and_grid_search.ipynb`](05_02_cross_validation_and_grid_search.ipynb)
- 입력 특성: `alcohol`, `sugar`, `pH`
- 타깃: `class`
- 모델: `DecisionTreeClassifier`
- 모델 선택: `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`

---

## 전체 흐름

```text
테스트 세트 분리
→ 훈련 세트에서 검증 세트 분리
→ 교차 검증으로 평균 성능 확인
→ GridSearchCV로 후보 전수 비교
→ RandomizedSearchCV로 일부 조합 탐색
→ 최적 모델 선택
→ 테스트 세트로 최종 평가
```

## 핵심 결과

| 단계 | 결과 |
|---|---:|
| 단일 검증 세트 정확도 | `0.8644` |
| 기본 5-폴드 평균 정확도 | `0.8553` |
| 섞은 10-폴드 평균 정확도 | `0.8574` |
| 확장 그리드 서치 최고 평균 | `0.8684` |
| 랜덤 서치 최고 평균 | `0.8695` |
| 랜덤 서치 최적 모델 테스트 정확도 | `0.8600` |
| `splitter='random'` 테스트 정확도 | `0.7869` |

---

## 1. 훈련·검증·테스트 세트

```python
train_input, test_input, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42
)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input,
    train_target,
    test_size=0.2,
    random_state=42
)
```

```text
전체 데이터
├─ 훈련 세트 80%
│  ├─ 훈련용 세트 80%
│  └─ 검증 세트 20%
└─ 테스트 세트 20%
```

- 훈련용 세트: 모델 학습
- 검증 세트: 모델과 하이퍼파라미터 선택
- 테스트 세트: 선택이 끝난 뒤 최종 성능 평가

> **이전 질문과 연결 — “테스트 정확도가 높으면 굳이 수정할 필요가 없지 않나?”**  
> 테스트 세트를 반복해서 확인하며 모델을 수정하면 테스트 세트에도 맞춘 셈이 됩니다. 수정 과정에는 검증 세트나 교차 검증을 사용하고 테스트 점수는 마지막에 확인해야 합니다.

제한 없는 결정 트리의 결과는 다음과 같습니다.

```text
훈련용 정확도: 0.9971
검증 정확도: 0.8644
```

훈련 데이터를 거의 외운 과대적합 상태입니다.

---

## 2. 교차 검증

검증 세트를 한 번만 나누면 어떤 샘플이 들어갔는지에 따라 점수가 달라질 수 있습니다. 교차 검증은 훈련 세트를 여러 조각으로 나누고 검증 역할을 번갈아 맡깁니다.

```python
scores = cross_validate(dt, train_input, train_target)
print(np.mean(scores['test_score']))
```

```text
기본 5-폴드 평균 정확도: 0.8553
```

`cross_validate()` 결과의 `test_score`는 최종 `test_input`의 점수가 아니라 각 폴드의 **검증 점수**입니다.

### 층화 K-폴드

```python
scores = cross_validate(
    dt,
    train_input,
    train_target,
    cv=StratifiedKFold()
)
```

분류 문제에서는 각 폴드의 클래스 비율을 유지하는 `StratifiedKFold`를 사용합니다. 기본 5-폴드 결과는 `0.8553`으로 같습니다.

```python
splitter = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)
```

```text
섞은 10-폴드 평균 정확도: 0.8574
```

> **이전 질문과 연결 — 훈련 점수와 테스트 점수의 차이는 어떻게 판단하나?**  
> 한 번 분리한 점수는 우연한 샘플 구성의 영향을 받을 수 있습니다. 여러 폴드의 평균을 사용하면 예상 성능을 더 안정적으로 판단할 수 있습니다.

---

## 3. 단일 하이퍼파라미터 그리드 서치

```python
params = {
    'min_impurity_decrease': [
        0.0001,
        0.0002,
        0.0003,
        0.0004,
        0.0005
    ]
}

gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    n_jobs=-1
)
gs.fit(train_input, train_target)
```

5개 후보를 각각 기본 5-폴드 교차 검증으로 비교합니다.

```text
최적 값: min_impurity_decrease=0.0001
후보별 평균 정확도:
[0.8682, 0.8645, 0.8649, 0.8678, 0.8676]
```

```python
dt = gs.best_estimator_
```

`best_estimator_`는 최적 조합을 선택한 뒤 전체 훈련 세트로 다시 학습한 모델입니다. 이 모델의 훈련 정확도는 `0.9615`입니다.

---

## 4. 여러 하이퍼파라미터 그리드 서치

```python
params = {
    'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
    'max_depth': range(5, 20, 1),
    'min_samples_split': range(2, 100, 10)
}
```

| 파라미터 | 역할 |
|---|---|
| `min_impurity_decrease` | 노드를 나누기 위한 최소 불순도 감소량 |
| `max_depth` | 트리의 최대 깊이 |
| `min_samples_split` | 노드를 나누기 위한 최소 샘플 수 |

후보 조합은 `9 × 15 × 10 = 1350개`입니다. 기본 5-폴드이므로 `6750회`의 학습·검증이 수행됩니다.

```text
최적 조합:
max_depth=14
min_impurity_decrease=0.0004
min_samples_split=12

최고 평균 검증 정확도: 0.8684
```

> **이전 질문과 연결 — “하이퍼파라미터는 내가 직접 설정해야 하나?”**  
> 사람이 탐색할 파라미터와 범위를 정하고, 그 범위 안의 비교는 검색 도구가 자동으로 수행합니다. 범위를 너무 좁게 잡으면 좋은 값을 놓치고, 너무 넓고 촘촘하게 잡으면 계산량이 커집니다.

---

## 5. 랜덤 서치

그리드 서치는 후보가 많아질수록 모든 조합을 검사해야 합니다. 랜덤 서치는 지정한 분포에서 일부 조합만 뽑습니다.

```python
params = {
    'min_impurity_decrease': uniform(0.0001, 0.001),
    'max_depth': randint(20, 50),
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 25)
}
```

- `randint(a, b)`: `a` 이상 `b` 미만 정수 추출
- `uniform(a, b)`: `a`부터 폭 `b`인 구간의 실수 추출

```python
rs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    n_iter=100,
    n_jobs=-1,
    random_state=42
)
rs.fit(train_input, train_target)
```

100개 조합을 기본 5-폴드로 비교하므로 `500회`의 학습·검증을 수행합니다.

```text
최적 조합:
max_depth=39
min_impurity_decrease≈0.000341
min_samples_leaf=7
min_samples_split=13

최고 평균 검증 정확도: 0.8695
최종 테스트 정확도: 0.8600
```

그리드 서치보다 훨씬 적은 조합을 확인하면서 비슷하거나 조금 높은 검증 점수를 얻었습니다.

---

## 6. 테스트 세트는 마지막에 평가

```python
dt = rs.best_estimator_
print(dt.score(test_input, test_target))
```

모델과 하이퍼파라미터 선택에는 훈련 데이터 내부의 교차 검증 결과만 사용합니다. 모든 선택이 끝난 후 테스트 세트를 한 번 평가한 결과는 `0.86`입니다.

---

## 7. `splitter='random'`

```python
DecisionTreeClassifier(
    splitter='random',
    random_state=42
)
```

기본 `splitter='best'`는 각 노드에서 가장 좋은 분할을 찾습니다. `splitter='random'`은 무작위 분할 후보를 사용합니다.

```text
최고 평균 검증 정확도: 0.8459
최종 테스트 정확도: 0.7869
```

이 실습에서는 기본 분할보다 성능이 낮았습니다.

> **이전 질문과 연결 — “`splitter='random'`으로 하면 엑스트라 트리인가?”**  
> 아닙니다. 이것은 한 개의 결정 트리가 무작위 분할을 사용하는 것입니다. 엑스트라 트리는 무작위성이 강한 트리를 여러 개 학습해 결과를 합치는 앙상블 모델입니다.

---

## 핵심 항목

| 항목 | 역할 |
|---|---|
| 검증 세트 | 모델과 하이퍼파라미터 선택 |
| 테스트 세트 | 최종 일반화 성능 평가 |
| `cross_validate()` | 여러 폴드의 훈련·검증 반복 |
| `StratifiedKFold` | 클래스 비율을 유지한 폴드 분할 |
| `GridSearchCV` | 지정한 모든 후보 조합 비교 |
| `RandomizedSearchCV` | 분포에서 일부 조합만 추출해 비교 |
| `best_params_` | 최적 하이퍼파라미터 |
| `best_estimator_` | 최적 값으로 전체 훈련 세트를 재학습한 모델 |
| `cv_results_` | 후보별 교차 검증 결과 |
| `n_jobs=-1` | 가능한 CPU 코어 모두 사용 |

---

## 정리

```text
모델 선택
→ 검증 세트 또는 교차 검증 사용

하이퍼파라미터 탐색
→ 후보가 적으면 그리드 서치
→ 범위가 넓으면 랜덤 서치

최종 평가
→ 모든 선택이 끝난 뒤 테스트 세트를 한 번 사용
```

핵심은 **하이퍼파라미터를 테스트 세트로 고르지 않는 것**입니다.
