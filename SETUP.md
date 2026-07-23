# 자동 진행률 설정 방법

압축을 푼 뒤 저장소 최상위에 아래 항목을 그대로 업로드합니다.

```text
README.md
.github/workflows/update-progress.yml
scripts/update_progress.py
notebooks/
requirements.txt
.gitignore
```

## 노트북 이름 규칙

```text
01_03_fish_classification.ipynb
02_01_train_test_set.ipynb
02_02_data_preprocessing.ipynb
03_01_knn_regression.ipynb
03_02_linear_regression.ipynb
```

위 파일을 `notebooks/` 아래 어느 폴더에 넣어도 자동으로 탐색합니다.

## 작동 확인

1. 파일을 GitHub에 업로드하고 커밋합니다.
2. 저장소 상단의 `Actions` 탭을 엽니다.
3. `Update README progress` 작업이 완료될 때까지 기다립니다.
4. README가 자동으로 수정되었는지 확인합니다.

처음 실행할 때 GitHub Actions가 저장소에 쓰지 못한다면:

1. `Settings`
2. `Actions`
3. `General`
4. `Workflow permissions`
5. `Read and write permissions`
6. `Save`

를 선택합니다.
