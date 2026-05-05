# 보조 테이블 EDA (Step 3-B-1)

메인 테이블 application_train: **307,511명** unique SK_ID_CURR

## 요약 표

| 테이블 | 행 | 열 | 메모리 | unique SK_ID_CURR | main 커버리지 | 결측률 q50 / max |
|---|---:|---:|---:|---:|---:|---:|
| `bureau` | 1,716,428 | 17 | 85.12MB | 305,811 | 85.69% | 0.0 / 0.7147 |
| `bureau_balance` | 27,299,925 | 3 | 156.21MB | - | - | 0.0 / 0.0 |
| `previous_application` | 1,670,214 | 37 | 146.56MB | 338,857 | 94.65% | 0.0 / 0.9964 |

## 테이블별 노트

### `bureau`
- shape: 1,716,428 × 17
- 메모리 (downcast 후): 85.12 MB
- main 커버리지: **85.69%** (263,491 / 307,511)
- unique SK_ID_BUREAU: 1,716,428
- dtype 분포: {'float32': 8, 'int32': 3, 'int16': 2, 'category': 1, 'int8': 1}
- 결측 분위수 q25/q50/q75/max: 0.0 / 0.0 / 0.1501 / 0.7147
- 결측 50%+ 컬럼 (2개): AMT_CREDIT_MAX_OVERDUE, AMT_ANNUITY

### `bureau_balance`
- shape: 27,299,925 × 3
- 메모리 (downcast 후): 156.21 MB
- bureau 대비 SK_ID_BUREAU 커버리지: **45.11%**
- dtype 분포: {'int32': 1, 'int8': 1, 'category': 1}
- 결측 분위수 q25/q50/q75/max: 0.0 / 0.0 / 0.0 / 0.0

### `previous_application`
- shape: 1,670,214 × 37
- 메모리 (downcast 후): 146.56 MB
- main 커버리지: **94.65%** (291,057 / 307,511)
- dtype 분포: {'float32': 15, 'int32': 3, 'int8': 2, 'category': 1, 'int16': 1}
- 결측 분위수 q25/q50/q75/max: 0.0 / 0.0 / 0.403 / 0.9964
- 결측 50%+ 컬럼 (4개): AMT_DOWN_PAYMENT, RATE_DOWN_PAYMENT, RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED
