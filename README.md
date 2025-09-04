# Detection5Invasive

5種の外来植物を検出するAIモデルを作動させるためのライブラリです。

検出できるのは以下の5種です。

- フランスギク
- ヒメジョオン
- キクイモ
- オオキンケイギク
- オオハンゴンソウ

---

## Requirements

- python 3.8.x (3.8.20)にて確認済み
- **PyToch**
  - PyTorchに関してはユーザーのPCのCUDAに応じて事前にインストールしてください
    - （公式 [PyTorch Install Guide](https://pytorch.org/get-started/locally/) を参照）
  - 通常の`ultralytics`および`YOLO`が使える環境であれば問題ないと思います
- `ultralytics-centerloss`
  - 本ライブラリの依存関係として自動的にインストールされます
- **注意**
  - ultralytics製の本家`ultralytics`は任意の仮想環境にインストールしないでください
  - 本家`ultralytics`とインストールされる`ultralytics-centerloss`が競合するため、`ultralytics`をインストールしている場合はアンインストールしてください

  ```bash
  pip uninstall ultralytics
  ```

---

## Install

1) 先に`pytorch`を環境に合わせてインストール（CUDAに合わせて選択）
2) 以下のコマンドを任意の仮想環境で実行し本ライブラリをインストール

    ```bash
    pip install "git+https://github.com/kato-seigou/Detection5Invasive.git@main"
    ```

---

## 使用方法

```python
from Detection5Invasive detection_pipeline

df = detection_pipeline(
    input_folder="path/to/images",
    process_folder="path/to/tmp",
    number=5,
    seed=42,
    model_path="path/to/model.pt",
    conf=0.7
)

print(df)

```
