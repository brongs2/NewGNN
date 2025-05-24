# NewGNN Project

## 개요
여러 GNN 모델(GCN, GCNII, ARGC)을 코어셋 데이터셋(Cora/CiteSeer/PubMed)에서 학습·평가하는 실험 코드입니다.

## 설치
```bash
git clone <https://github.com/brongs2/NewGNN.git>
cd NewGNN

# 의존성 설치
pip install -r requirements-base.txt
pip install -r requirements-pyg.txt

# 데이터 다운로드 및 학습 실행
python src/train.py --model ARGC --dataset Cora --layers 32 --epochs 200 --lr 0.001
