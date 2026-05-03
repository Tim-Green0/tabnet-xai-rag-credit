# Home Credit Default Risk 데이터 다운로드 가이드

이 프로젝트는 Kaggle의 [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) 메인 테이블만 사용한다. 두 가지 방법 중 편한 쪽을 골라.

---

## 방법 A. Kaggle API (자동, 권장)

### A-1. Kaggle 계정 + 대회 규칙 동의

1. https://www.kaggle.com 가서 로그인 (없으면 가입)
2. https://www.kaggle.com/competitions/home-credit-default-risk **접속**
3. 우측 상단 **"Join Competition"** 또는 **"Late Submission"** 버튼 클릭 → 대회 규칙(Rules) 동의
   - **이 단계 안 하면 API로 다운로드 시 403 에러 남.**

### A-2. API 토큰 발급

1. https://www.kaggle.com/settings (또는 우측 상단 프로필 → Account)
2. 페이지 중간 **"API"** 섹션 → **"Create New Token"** 클릭
3. `kaggle.json` 파일이 자동 다운로드됨
4. 그 파일을 `C:\Users\ws059\.kaggle\kaggle.json` 위치로 옮김
   ```bash
   mkdir -p /c/Users/ws059/.kaggle/
   mv ~/Downloads/kaggle.json /c/Users/ws059/.kaggle/
   ```

### A-3. 다운로드

내가 venv에 `kaggle` 패키지를 설치해두면, 사용자가 토큰만 위 경로에 두고 알려주면:
```bash
cd /d/paper
.venv/Scripts/kaggle.exe competitions download -c home-credit-default-risk -f application_train.csv -p data/home_credit/
.venv/Scripts/python.exe -c "import zipfile; zipfile.ZipFile(r'data/home_credit/application_train.csv.zip').extractall('data/home_credit/')"
```
이걸 내가 대신 실행하면 끝. **`kaggle.json` 위치를 잡아두면 알려줘** — 내가 위 명령 실행해줄게.

> 참고: 메인 파일만 받으면 약 38 MB(zip) → 압축 풀면 약 166 MB.

---

## 방법 B. 수동 다운로드 (간단, 토큰 발급 귀찮을 때)

1. https://www.kaggle.com/competitions/home-credit-default-risk/data 접속
2. 대회 규칙 동의 (안 했으면)
3. 페이지 아래 **Data Sources** 섹션에서 `application_train.csv` 항목의 다운로드 아이콘(↓) 클릭
   - 또는 페이지 하단 **"Download All"** 버튼(전체 zip ~700MB)
4. 받은 `application_train.csv` (또는 `home-credit-default-risk.zip`)을 다음 위치로 이동:
   ```
   D:\paper\data\home_credit\application_train.csv
   ```
5. 옮긴 후 나에게 "데이터 옮겼어"라고 알려주면 EDA 시작.

> **메인 테이블만** 필요하니 전체 zip을 받았다면 application_train.csv 하나만 옮겨도 충분 (다른 파일은 future work).

---

## 검증 (어느 방법이든 끝난 후 내가 실행)

```python
import pandas as pd
df = pd.read_csv('data/home_credit/application_train.csv')
print(df.shape)         # 예상: (307511, 122)
print(df['TARGET'].mean())  # 예상: 0.0807 (약 8% 부도율)
```
