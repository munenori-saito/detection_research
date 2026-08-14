## 仮想環境を作成する

#　Windowsの場合

# 仮想環境の作成
python -m venv .venv

# 有効化
.\.venv\Scripts\Activate.ps1

#　Macの場合

# 仮想環境の作成
python3 -m venv .venv

# 有効化
source .venv/bin/activate

pip install -r requirements.txt

python detect.py