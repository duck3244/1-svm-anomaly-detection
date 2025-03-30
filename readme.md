# One-Class SVM for Anomaly Detection

이 프로젝트는 One-Class SVM을 이용한 이상 탐지(Anomaly Detection) 구현 코드입니다.

## 파일 구조

- `main.py`: 메인 실행 파일
- `data_loader.py`: 데이터 로딩 및 전처리 기능
- `anomaly_generator.py`: 이상치 데이터 생성 기능
- `one_class_svm.py`: One-Class SVM 모델 훈련 및 평가 기능
- `utils.py`: 결과 시각화 및 유틸리티 함수
- `requirements.txt`: 필요한 라이브러리 목록

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python main.py
```

## 기능 설명

### 1. 데이터 로딩 및 준비
- `train_FD001.txt`와 `test_FD001.txt` 파일에서 데이터를 로드
- 데이터 전처리 및 특징 추출

### 2. 이상치 데이터 생성
- 정상 데이터의 최대/최소값을 기반으로 이상치 데이터 생성
- 최대값 이상 및 최소값 이하의 데이터 생성

### 3. One-Class SVM 모델 훈련
- RBF 커널을 사용한 One-Class SVM 모델 훈련
- 모델 저장 및 로드 기능

### 4. 결과 시각화
- 분류 결과 시각화
- 이상치 탐지 경계 시각화
- Precision-Recall 곡선 생성

## 출력 파일
- `Distribution.png`: 데이터 분포 시각화
- `One-Class SVM.png`: SVM 경계 및 이상치 탐지 결과
- `Precision-Recall_Curve.png`: 모델 성능 평가
- `OneClassSVMModel.pickle`: 훈련된 모델 저장 파일
