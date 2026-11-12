昨年、新たな試みとして、研究を始める直前のB4に向けて、一話完結のPython輪講を行いました。<br>
本資料は、それを大幅にアップデートしたものです。<br>
Powerpointだと、短いプログラムしか載せられないので、GROWIの記事として作成することにしました。

# 用語
| 言葉             | 説明                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------- |
| PyTorch          | 深層学習を実装するためのフレームワーク。他にはTensorFlowも有名だが、神谷研は（おそらく）全員PyTorchユーザ。 |
| テンソル(Tensor) | PyTorchで使われる配列構造。                                                                                 |
| テンソルの大きさ | テンソル`x`に対して`x.shape`によって取得できる属性。2次元画像では、通常`(B, C, H, W)`となる。               |

# 画像処理タスク
本資料では、直感的に分かりやすい二つのタスクに注目する。<br>
他のタスクでも、基本的なところは共通しているので、本資料の内容が理解の助けになるはずだ。

## 分類(Classification)
画像における深層学習は、分類から始まった。<br>
先駆的な研究例として、Fukushima(1988)のNeocognitron[^1]、Lecun et al.(1998)のLeNet-5[^2]による手書き数字認識が挙げられる。<br>
これは、各数字に対して0～9の10クラス分類を行っているといえる。<br>
FukushimaとLecun et al.では少し推論機構が異なるが、現代的な分類タスクでは、画像が各クラスに属する確率を予測する。<br>
つまり、`0`が`0.05(5%)`、`1`が`0.1(10%)`といったように、総和が`1`となる10次元のベクトルを出力する。<br>
そして、その中で最も確率の高いクラスが、モデルの予測結果として扱われる。

## セグメンテーション(Segmentation)
上述の分類が**画像全体の分類タスク**だったのに対して、セグメンテーションは**ピクセルレベルの分類問題**といえる。<br>
入力を`(B, C, H, W)`、対象のクラス数を`L`とすると、出力は`(B, L, H, W)`となり、各画素が`L`次元の確率ベクトルを持っていると解釈できる。<br>
そして、各画素で確率が最大のクラスを取ってくることで、最終的なセグメンテーション結果が得られる。

# 深層学習の全体像
画像処理のニューラルネットワークは、三つの基本構造「入力」「バックボーン（Backbone：特徴抽出器）」「出力（Task Head）」で構成される。<br>
入力・バックボーン・出力をすべてひっくるめたものは、一般的にモデル（Model）と呼ばれる。<br>
それぞれの役割は、以下のとおりである。

## 入力
データを読み込んで、バックボーンに渡す処理を行う。

## バックボーン
入力されたデータから特徴量を抽出する。<br>
バックボーンは、主に以下の部品から構成される。

### 活性化関数
モデルに非線形性を与える。<br>
その他、特定の目的のために使われる関数もある。

### 残差接続
ResNet[^3]で導入された機構で、モデル内での情報伝達を強化することにより、勾配消失問題を解決した。

### 正規化
特徴量をスケーリングすることにより、数値的な安定性を向上させる。

### CNN(Convolutional Neural Network)
最も基本的な特徴抽出機構で、画像分野の深層学習技術においてブレイクスルーを起こした。

### Transformer
CNNの次にブレイクスルーを起こした特徴抽出機構で、現在のスタンダードと言っても過言ではないレベルの高性能を誇る。

## 出力
CNNやTransformerなど、バックボーンによって抽出された特徴量をもとに、タスクを行う。

## フィードバック
### そもそも、深層学習モデルとは何か
関数$f(x)=2x+1$を考える。<br>
$f(1)=3$、$f(2)=5$という要領で、$x$に数値を入力すると、それに対する出力が得られる。

深層学習も、入力・バックボーン・出力のそれぞれが複雑になっているだけで、構造は関数とまったく同じである。<br>
データを与えると、モデルがそれを処理し、結果が出力されるという一連の流れは、関数そのものだ。<br>
実際、画像輪講Ⅱで「深層学習は複雑な関数である」ということの意味が理解できただろう。

### 損失関数
誤差逆伝播法というものを画像輪講Ⅱで勉強した。<br>
これは、勾配消失問題をうまく回避しつつ、モデルのパラメータを調整することができるクリティカルな理論だった。<br>
損失のフィードバックについて、分かりやすい例とともに振り返る。

人間の身長と体重の関係は一次関数$f(x)=ax+b$でモデリングできる、という仮説を立てたとする。<br>
このとき、適当な人数(e.g., 1,000人)の身長・体重のデータを集めてきて最小二乗法を使えば、モデリングは達成される。

深層学習の文脈に近づけていうと、$f(x)=ax+b$は、身長$x$が入力されたとき、その真の体重$y$に対する予測値$f(x)$を出力する関数だと解釈できる。<br>
しかし、当然、身長と体重の関係を一次関数で完全にモデリングできるわけではない。<br>
誤差が小さい、すなわち良いモデルを作るためには、まず、真値と予測値の誤差を定義する必要がある。<br>
ここでは、二乗誤差$(y-f(x))^2$とする。<br>
そして、その総和が手元の1,000人分のデータで小さくなるように、パラメータ$a$と$b$を少しだけ調整する。<br>
予測、誤差計算、パラメータ調整を繰り返すことで、徐々に誤差が小さくなっていく。

深層学習でも、基本的な構造は同じだ。<br>
画像分類のモデルを考えよう。<br>
`犬`の画像を入力して、モデルが`猫`と予測したとする。<br>
損失（＝誤差）を数値化して、モデルにフィードバックすることで、その入力画像に対して`犬`と出力できるようにパラメータを調整する。<br>
このサイクル`Input -> Model -> Output -> Loss Function -> Feedback`を、たくさんのデータを用いて繰り返すことで、分類タスクをこなせるようになる。

# 入力
## データセット
教師あり学習では、入力と正解のペアをたくさん集めたものがデータセットである。<br>
[U-Net](https://github.com/milesial/Pytorch-UNet)では、データセットクラスは、次のように実装されている。

> なお、可読性のため、必要最低限のところだけを残して、プログラムを簡略化している。<br>気になる人は、元のプログラムを見てみよう。

- ポイント1：PyTorchの`Dataset`を継承している。
- ポイント2：`__len__`をオーバーライドしている。これでデータ数を返す。
- ポイント3：`__getitem__`をオーバーライドしている。これで実際にデータを読み込んで返す。

```py:dataset.py
from pathlib import Path
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        # ディレクトリの中にあるファイルを検索して、self.idsにリストとして保存

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # 画像の前処理
        if is_mask:
            # mask(セグメンテーションの正解データ)の処理
            return mask

        else:
            # 画像の処理
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]   # インデックスに対応するファイル名を取得

        # imgに画像を読み込んでpreprocessで処理
        # maskに正解データを読み込んでpreprocessで処理

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
```

### 学習・検証・テスト
そもそもの目的を思い出そう。<br>
私たちはデータセットを用意して、場合によっては労力のかかるアノテーションをして、高価なGPUを使ってモデルを学習させて、何をしたいのだろうか？<br>
学習に使ったデータセットだけで推論させて遊びたいわけでは、ないはずだ。<br>
学習に使っていない未知データに対して、分類やセグメンテーションなどのタスクを行いたいのだった。

しかし、ちょっとした問題がある。<br>
既知データでうまく推論ができたとしても、未知データでもうまくいくとは、かぎらないのである。<br>
未知データに対するモデルの推論能力を、**汎化性能**という。<br>
モデルを学習させるとき、既知データでの性能を高めつつ、実用のことを考えて汎化性能も確かめる必要がある。<br>
そのため、データセットのすべてを学習に費やすのではなく、役割ごとに分けるのが一般的だ。

| 役割             | 説明                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| 学習(Train)      | モデルのパラメータを調整する。                                                                 |
| 検証(Validation) | 学習中に汎化性能を確かめる。テストに使うモデルを、検証での精度が最も高いものとすることがある。 |
| テスト(Test)     | 学習後に汎化性能を確かめる。                                                                   |

データセットを役割ごとに分けるときには、**データが完全に独立する**ように細心の注意を払おう。<br>
例えば、学習に使われたデータの一部がテストに混ざっているという状況は、あってはいけない。<br>
最近の授業で説明されたこと（＝学習）が、期末試験（＝テスト）にそのまま出題されたら、生徒の学力（＝モデルの汎化性能）を正確に測定することはできない。

ちなみに、学習・検証・テストではなく、検証を省略することも多々ある。<br>
この場合、テストに用いるモデルは、学習が終了したときのものにすると良い。

### データローダ(Data Loader)
PyTorchにデータローダのクラスが実装されている。<br>
使用イメージは、以下のとおりだ。

```py:data_loader.py
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# 自分で実装したクラスをimportする
from hoge import BasicDataset

# 環境に応じてパスを書きかえる
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')

def train_model():
    # データセットを実体化する
    dataset = BasicDataset(dir_img, dir_mask)

    # Train/Testに分割する
    # ここではランダムだが、再現性のために、事前に分け方を決めておくのも良い
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # データローダを実体化する
    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)
```

## データ拡張(Data Augmentation)
たくさんのデータを用意して学習に使わないと、モデルの汎化性能は向上しない。<br>
かぎられた資源を最大限に活用するために、データ拡張を行うのが一般的である。<br>
これは、学習用の画像データに回転・反転・平行移動などの幾何変換を加えることで、疑似的にデータ数を増やす処理だ。<br>
自作しても良いし、手軽に試したいのであれば[torchvision.transforms](https://docs.pytorch.org/vision/0.8/transforms.html)が便利だろう。

### TTA(Test Time Augmentation)
データ拡張といえば、基本的には学習時のことを指すが、テスト時にもデータ拡張を行うTTAという手法がある。<br>
一枚の画像に対して一回の推論を行うのが、普通の推論である。<br>
一方で、TTAでは、データ拡張によって$N$枚（e.g., 原画像、上下反転、左右反転、上下左右反転の計四枚）に増やしてそれぞれ推論を行い、その結果を統合して最終的な出力を得る。<br>
これにより、多くの場合、モデルの性能を底上げすることができる。

# バックボーン
## 活性化関数
### Tanh
入力を`[-1, 1]`にスケーリングする関数である。<br>
深層学習の黎明期に使われていたが、絶対値の大きい値も$|1|$以内に抑え込まれてしまうため、今では採用される場面は限定的だ。

$$Tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

![Tanh.png](/attachment/6a03cdede4ba3f4e1317d217)

### ReLU(Rectified Linear Unit)
最も一般的な活性化関数である。<br>
計算は速いが、$x=0$で微分不可能なので、それを改善するための関数が多数提案されている。

$$ReLU(x)=max(0, x)$$

![ReLU.png](/attachment/6a03ce49e4ba3f4e1317d235)

### GELU(Gaussian Error Linear Unit)
標準正規分布の累積分布関数$\Phi(x)$を用いて計算される関数であり、全領域で微分可能。<br>
Transformer系のモデルで特に好んで使われ、ReLUより計算は遅いが、性能が良いことで知られている。

$$GELU(x)=x*\Phi(x)$$

![GELU.png](/attachment/6a03cebde4ba3f4e1317d262)

### Sigmoid
入力$x$の**各要素**が`[0, 1]`に収まるようにスケーリングする。

$$Sigmoid(x)=\frac{1}{1+e^{-x}}$$

![Sigmoid.png](/attachment/6a03cf1be4ba3f4e1317d280)

### Softmax
名前のとおり、柔らかい最大値関数である。<br>
入力$x$に対して、単純なmaxは最大値を`1`、それ以外を`0`にするものといえるが、Softmaxは値の大小関係を残したまま**各要素**が`[0, 1]`にスケーリングされ、かつ**全要素の合計**が`1`になる。<br>
この性質から、TransformerのSelf-attentionや損失関数の中で採用される。

$$Softmax(x_i)=\frac{e^{x_i}}{\sum_{j}e^{x_j}}$$

## 残差接続(Residual Connection)
CNNやTransformerなどの構成要素において、入力$x$から特徴を抽出する層(Layer)を関数$F(x)$で表す。<br>
実際に学習される関数を$H(x)$とすると、普通は$H(x)=F(x)$となる。<br>
一方で、残差接続では$H(x)=F(x)+x$となり、入力$x$を加えたものが学習される。

![residual.png](/attachment/6a03dfbbe4ba3f4e1317d379)

[Transformer](https://github.com/lucidrains/vit-pytorch)を見てみると、次のように実装されている。

- ポイント1：`nn.Module`を継承してモデルの構成要素を実装している。
- ポイント2：`forward`をオーバーライドしている。これで入力`x`を処理して返す。なお、`forward`は演算子`()`を有効にするものである。

```py:residual_transformer.py
import torch.nn as nn

# 以下の二つを実装済みとする
# class Attention
# class FeedForward

# めっちゃ簡略化してます
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = FeedForward()
    
    def forward(self, x):
        x = x + self.attn(x)   # residual
        x = x + self.ff(x)     # residual
        return x
```

`residual_transformer.py`の場合は、残差接続の途中で関数が一つしかないので、特に気にすることはない。<br>
途中の関数が`f(x)`、`g(x)`のように複数あるとき、`x = x + g(f(x))`と書いても良いが、以下のような表記のほうが多い印象。

```py:residual_fg.py
import torch.nn as nn

# 以下の二つを実装済みとする
# class f
# class g

class Hoge(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = f()
        self.g = g()
    
    def forward(self, x):
        shortcut = x       # 元の特徴を保存。必要に応じてx.clone()にする
        x = self.f(x)
        x = self.g(x)
        x = x + shortcut   # 残差接続
        return x
```

## 正規化
Wu et al.[^4]から図を引用する。<br>

![normalization.png](/attachment/6a03e18be4ba3f4e1317d504)

### バッチ正規化`nn.BatchNorm2d`
バッチ全体を通して、チャネルごとに正規化を行う。<br>
主にCNN系のモデルで使われる。<br>
バッチサイズが小さいときは統計量が不安定になるため、注意が必要。

### レイヤ正規化`nn.LayerNorm`
チャネル全体を通して、データごとに正規化を行う。<br>
主にTransformer系のモデルで使われる。

### インスタンス正規化`nn.InstanceNorm2d`
データごと、かつチャネルごとに正規化を行う。<br>
画像が持つコントラストの情報をリセットする役割があり、主に敵対的生成ネットワークで使われる。

### グループ正規化`nn.GroupNorm`
チャネルをいくつかのグループに分割して、グループごとに正規化を行う。<br>
レイヤ正規化とインスタンス正規化の中間的な存在で、バッチサイズを下げざるをえないときにバッチ正規化の代わりに使われることがある。

## CNN
### 畳み込み(Convolution)
`3*3`などのフィルタの重みを学習する。<br>
[U-Net](https://github.com/milesial/Pytorch-UNet)では、以下のように実装されている。

- ポイント1：モデルの構成要素には、とにかく`nn.Module`を使う！
- ポイント2：畳み込み（特徴抽出）、正規化、活性化関数という順番で、基本に忠実な王道の処理を採用している。
- ポイント3：隠れ層（`mid_channels`、他には`hidden_channels`などの表記もあり）によって表現力の向上を図っている。
- ポイント4：`nn.Sequential`で直列の処理をまとめて、分かりやすくしている。

```py:conv_layer.py
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

### 受容野(Receptive Field)
画像処理分野の深層学習において、受容野という重要な概念がある。<br>
上記のクラス`DoubleConv`では、`nn.Conv2d`を用いて畳み込みを行っている。<br>
注目してほしいのは、`kernel_size=3`というパラメータである。<br>
この畳み込みフィルタは、一回の演算でたった`3*3`の範囲しか見ていない。<br>
入力テンソルの大きさを`(B, 3, 512, 512)`とすると、`3*3`は、テンソルのごくわずかな一部にすぎない。

受容野を広げるために、`kernel_size=5`、あるいはもっと大きくするのも一案である。<br>
しかし、計算量が2乗のペースで増えてしまうため、実用的ではない。

そこで、小手先のテクニックとして、`kernel_size=3`から`5`に増やす代わりに、`3*3`のフィルタを二つ用意して直列につなげるという技法が知られている。<br>
一回目の畳み込みでは、`3*3`の範囲しか見ていないが、その結果に対してもう一回`3*3`の畳み込みを行うと、間接的に`5*5`の範囲を見ることになるからだ。<br>
さらに、`kernel_size=5`では、一回の畳み込みで$5^2=25$回の計算が必要だったのに対して、`kernel_size=3`を二つ直列につなげたものでは、$3^2*2=18$回の計算に抑えることができている。<br>
つまり、`DoubleConv`では、計算量を抑えつつ受容野を広げるという、実践的な知見が反映された実装になっているといえる。

### プーリング(Pooling)
とはいえ、`kernel_size=3`のフィルタを直列にたくさんつなげて受容野を広げるというのは、非効率的だ。<br>
より賢い方法として、プーリングによって特徴マップの空間的な解像度を小さくするのが一般的である。<br>
具体的には、[U-Net](https://github.com/milesial/Pytorch-UNet)では以下のように実装されている。

```py:down_sampling.py
import torch.nn as nn

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
```

特徴マップの解像度を下げる目的では、通常、MaxPoolingが使われる。<br>
これには、最も反応の強いものを残しておくことで、解像度を下げても効果的な特徴抽出を可能にするという狙いがある。<br>
その他、特徴抽出をサポートするモジュールでは、情報を要約するためにAveragePoolingもよく採用される。

### 分類モデル
分類モデルの例として、最初の現代的なCNNである[AlexNet](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)を紹介する。<br>
もっと新しいモデルだと構造が複雑になるが、AlexNetは、ここまでの説明で十分に理解できるレベルである。

- ポイント1：`nn.Conv2d`、`nn.ReLU`はセットで、解像度を落とすときは`nn.MaxPool2d`を使っている。
- ポイント2：正規化で学習を安定させる思想は、まだ導入されていない。
- ポイント3：チャネル数を段階的に`3 -> 64 -> 192 -> 384`と引き上げて、深い層で豊かな特徴表現を学習している。なお、最初の畳み込み`3 -> 64`は、生の特徴量`RGB`などを高度な特徴空間に移すという意味で`Embedding`と呼ばれることがある。

```py:alexnet_convs.py
import torch.nn as nn

# 簡略化してます
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   # 大きいカーネルなので計算コストは高い
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        #
        # この後、出力用の処理を行います（詳細は出力の章へ）
        #
        return x
```

### セグメンテーションモデル
#### アップサンプリング(Upsampling)
分類が`画像全体`の分類問題なのに対して、セグメンテーションは`ピクセルレベル`の分類問題であった。<br>
入力を`(B, C, H, W)`、対象のクラス数を`N`とすると、出力は`(B, N, H, W)`となる必要がある。

しかし、ここまでの内容だけを踏まえて考えると、不都合なことが起こる。<br>
受容野を広げるという要請によって、特徴抽出の過程で、特徴マップは縮小されているのだ。<br>
入力の解像度が`(512, 512)`だったとすると、最終的には`(32, 32)`などの解像度になっている。<br>
これでは、出力で求められている大きさと合っていない。

そこで、プーリングと対になる処理として、アップサンプリングが必要となる。<br>
これにはさまざまな手段があるが、よく見るのは、`import torch.nn.functional as F`として、`F.interpolate`でバイリニア補間を行う方法だ。<br>
ちなみに、Poolingの反対がUnpooling、Sampling（あるいは、Downsampling）の反対がUpsamplingで、これらの言葉はどれも等しく使われている印象である。

#### スキップ接続(Skip-connection)
アップサンプリングで`(32, 32)`を一気に`(512, 512)`に持っていくことは可能だが、実用的には効果が薄い。<br>
そのため、`(32, 32) -> (64, 64) -> ...`と段階を踏んで、`nn.Conv2d`を使いながら特徴マップを拡大する。<br>
しかし、`(32, 32)`という小さな特徴マップから、元の解像度の情報を復元するのが難しいというのは、直感的にも納得できる。

そこで導入されたのがスキップ接続という技術であり、セグメンテーションの発展を支えてきたU-Net系のモデルにおいて必要不可欠だ。<br>
模式図を以下に示す。（図中の数字は解像度を表している）

![unet_architecture.png](/attachment/6a03ea47e4ba3f4e1317d742)

| 構成要素            | 説明                                                                                                                              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| エンコーダ(Encoder) | `512 -> ... -> 32`の部分。解像度を下げながら特徴を抽出する。最後の層（`32*32`のところ）は、モデルによってあったりなかったりする。 |
| デコーダ(Decoder)   | `32 -> ... -> 512`の部分。解像度を上げながら特徴を洗練させる。                                                                    |
| スキップ接続        | 点線の矢印。同じ解像度の特徴マップをエンコーダから持ってきて統合することで、高周波成分を補う。                                    |
```py:up_sampling.py
import torch.nn as nn

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        # アップサンプリング
        x1 = self.up(x1)

        # サイズ調整
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # スキップ接続
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

#### U-Net
ようやく、U-Netの全体像を説明できる知識が出揃った。<br>
今までに説明したクラスを組み合わせるだけで、U-Netの特徴抽出機構を作ることができる。

- ポイント1：AlexNetと同じで、`Down`や`Up`に渡されているのは、チャネル数である。一瞬、特徴マップの大きさに見えるかもしれないので、注意すること。
- ポイント2：`nn.Sequential`が使われていない。これは、1入力1出力がずっと守られているときに使えるものだ。U-Netでは、デコーダで2入力1出力になるため、直列処理が自力で書かれている。
- ポイント3：入力のチャネル数が変数`n_channels`になっている。AlexNetでは`3`に固定されていたが、変数化することで自由度が上がっている。

```py:unet_convs.py
import torch.nn as nn
from hoge import DoubleConv, Down, Up

# 簡略化してます
# コメントはチャネル数の推移を表す
class UNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2
        self.down4 = (Down(512, 1024 // factor))   # 512 -> 512
        self.up1 = (Up(1024, 512 // factor))       # 512*2(skip-connectionのtorch.catがあるため) -> 256
        self.up2 = (Up(512, 256 // factor))        # 256*2 -> 128
        self.up3 = (Up(256, 128 // factor))        # 128*2 -> 64
        self.up4 = (Up(128, 64))                   # 64*2 -> 64

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #
        # この後、出力用の処理を行います（詳細は出力の章へ）
        #
        return x
```

## Transformer
### Attention Is All You Need
Transformer[^5]は、元々は自然言語処理で画期的な成果を上げたモデルだ。<br>
分かりやすい例として、Alammar[^6]の記事を紹介する。

`The animal didn't cross the street because it was too tired.`という文を考えよう。<br>
この中の`it`は、何を指すだろうか？<br>
人間には、`animal`だということが簡単に分かるが、機械にとってはそうではない。

Transformerの根幹をなす処理、Self-attentionでは、注目している文を単語レベルに分解して、それぞれの関連性を学習する。<br>
これによって、`it`は`animal`との関連性が高いということが学習される。<br>
ちなみに、この画像では表現されていないが、`it`は疲れているので、`tired`との関連性が高いことも学習される。

![self_attention.png](/attachment/6a03ed93e4ba3f4e1317d8ab)

### モデル構造
Transformerは、特徴抽出を担うBlockを基本単位として、それを直列につなげて構成される。<br>
Blockの構成要素は、以下のとおりである。

| 構成要素                    | 説明                                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Self-attention              | Transformerの基幹処理である。                                                                                |
| MLP(Multi-Layer Perceptron) | より一般的にはFFN(Feed Forward Network)といわれるが、中身は隠れ層を一つ持ったMLPとして実装されることが多い。 |
| 正規化                      | 数値的安定性のために導入される。                                                                             |
| 残差接続                    | 勾配消失問題を防ぐために導入される。                                                                         |

![block.png](/attachment/6a03ee43e4ba3f4e1317d8f7)

### Vision Transformer
Transformerの機構を画像処理に応用したものが、ViT(Vision Transformer)[^7]である。<br>
自然言語処理のTransformerが文を単語に分解したように、画像処理では画像をパッチ（`16*16`などの正方形領域）に分割して、パッチ間の関連性を学習する。<br>
CNNが畳み込みとプーリングを繰り返して少しずつ受容野を広げるのに対して、Transformerでは特徴抽出の初期段階から大きい受容野を持つことが可能だ。

![vit.png](/attachment/6a03eee1e4ba3f4e1317d933)

1. 画像をパッチ(Patch)に分割する。
2. 各パッチを、トークン(Token)と呼ばれる固定長のベクトルに変換する。
3. Self-attentionでは位置情報を扱えないので、原画像における位置情報を埋め込む。(Positional Encoding)
4. トークンに対してSelf-attentionを適用する。

以上の処理は、[例えばこのように](https://github.com/lucidrains/vit-pytorch)実装されている。

- ポイント1：正規化を前に置くか(Pre-norm)、後に置くか(Post-norm)は、モデルの設計思想によって異なる。この実装例ではPre-normが採用されている。
- ポイント2：活性化関数には`nn.GELU`、正規化には`nn.LayerNorm`が採用されている。いずれもTransformerの基本である。
- ポイント3：Self-attentionとFFNをそれぞれ独立のクラスとして実装して、それらを使ってBlockを構成している。

```py:transformer_block.py
import torch.nn as nn
from einops import rearrange

# Feed Forward Network
# 標準的なMLP
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Self-attention
# 処理の詳細は、今は理解する必要はない
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)   # Pre-norm

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Block
# 基本単位である Attention+FeedForward をdepthの分だけ直列につなげている
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # residual
            x = ff(x) + x     # residual

        return self.norm(x)   # この実装では出力にも正規化をかけている
```

### 分類モデル
[上記のTransformer](https://github.com/lucidrains/vit-pytorch)が分類用の実装なので、そのまま引用していく。<br>
細かいところは省略しているが、パッチ分割、トークンへの変換、Positional Encoding、Self-attentionが順番に行われていることが分かる。

```py:transformer_attn.py
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # パッチ分割の準備
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # パッチ分割
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # クラス分類用トークンの準備
        num_cls_tokens = 1 if pool == 'cls' else 0
        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))

        # Positional Encodingの準備
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # メインの特徴抽出器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)
        #
        # この後、出力用の処理を行います（詳細は出力の章へ）
        #
        return x
```

### セグメンテーションモデル
Transformer系のセグメンテーションモデルで分かりやすいものがなかったので、木原が研究で実装した[MetaFormer](https://github.com/sail-sg/metaformer)ベースのモデルを紹介する。<br>
~~なお、実装が正しいとはかぎらない。~~

```py:metaformer_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from metaformer_baselines import Downsampling, MetaFormerBlock, Pooling, SepConv, Attention, LayerNormWithoutBias, Mlp

class Upsampling(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # [B, H, W, C] のままチャネル方向の線形結合を行う
        self.fusion = nn.Linear(in_channels + skip_channels, out_channels)

        # 結合後の安定化
        self.norm = LayerNormWithoutBias(out_channels)

    def forward(self, x, skip):
        # x: [B, H, W, C], skip: [B, H, W, C]
        
        # Upsampling (PyTorchのinterpolateを使うために次元を入れ替える)
        x = x.permute(0, 3, 1, 2)
        target_size = (skip.shape[1], skip.shape[2])
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)

        # Skip-connectionでエンコーダからの情報を統合
        out = torch.cat([x, skip], dim=-1)
        
        # チャネル数の圧縮と正規化
        out = self.fusion(out)
        out = self.norm(out)
        return out

MIXER_MAP = {
    "Pooling": Pooling,
    "SepConv": SepConv,
    "Attention": Attention,
    "Identity": nn.Identity
}

class UMetaFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_token_mixers=("SepConv", "SepConv", "Attention", "Attention", "Attention"),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(256, 128, 64, 64),
        dec_token_mixers=("Attention", "Attention", "SepConv", "SepConv"),
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.num_enc_stages = len(enc_depths)
        self.num_dec_stages = len(dec_depths)

        # --- 1. Encoderの構築 ---
        self.downsamples = nn.ModuleList()
        self.enc_stages = nn.ModuleList()
        
        # DropPathの確率をネットワーク全体で徐々に上げる
        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        
        in_c = in_channels
        cur_dp = 0
        for i in range(self.num_enc_stages):
            out_c = enc_channels[i]
            # 最初だけストライド4(パッチ化)、以降はストライド2でダウンサンプリング
            stride = 4 if i == 0 else 2
            padding = 2 if i == 0 else 1
            kernel_size = 7 if i == 0 else 3
            pre_permute = False if i == 0 else True
            
            # ダウンサンプリング
            self.downsamples.append(
                Downsampling(
                    in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding,
                    pre_norm=partial(LayerNormWithoutBias, eps=1e-6) if i > 0 else None,
                    pre_permute=pre_permute
                )
            )

            # Mixerの種類を取得
            # MetaFormerは、Transformerが成功した理由について、Self-attentionではなく、
            # その構造(Mixer -> Residual -> MLP -> Residual)が優れているからだと主張した。
            # そのため、Mixerを切りかえられるような設計になっている。
            # __init__の引数で指定されている既定値のMixersは、
            # 解像度が高い層でConv、低い層でSelf-attentionを行うことで、
            # CNNとTransformerの性質を最大限に活かす構成になっている。
            TokenMixer = MIXER_MAP[enc_token_mixers[i]]
            
            # メインの特徴抽出
            stage_blocks = []
            for j in range(enc_depths[i]):
                stage_blocks.append(
                    MetaFormerBlock(
                        dim=out_c,
                        token_mixer=TokenMixer,
                        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                        drop_path=enc_dp_rates[cur_dp + j],
                    )
                )
            self.enc_stages.append(nn.Sequential(*stage_blocks))
            in_c = out_c
            cur_dp += enc_depths[i]

        # --- 2. Decoderの構築 ---
        self.upsamples = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        
        dec_dp_rates = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(dec_depths))]
        
        # ボトルネック(Stage 5)のチャネル数からスタート
        dec_in_c = enc_channels[-1] 
        cur_dp = 0
        
        # デコーダは下から上へ(Stage 4 -> Stage 1)
        for i in range(self.num_dec_stages):
            # 結合するSkip Connectionのチャネル数 (エンコーダの逆順: Stage 4, 3, 2, 1)
            skip_c = enc_channels[-(i + 2)] 
            out_c = dec_channels[i]
            
            # アップサンプリング
            self.upsamples.append(
                Upsampling(in_channels=dec_in_c, skip_channels=skip_c, out_channels=out_c)
            )

            TokenMixer = MIXER_MAP[dec_token_mixers[i]]
            
            # メインの特徴抽出
            stage_blocks = []
            for j in range(dec_depths[i]):
                stage_blocks.append(
                    MetaFormerBlock(
                        dim=out_c,
                        token_mixer=TokenMixer,
                        norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                        drop_path=dec_dp_rates[cur_dp + j],
                    )
                )
            self.dec_stages.append(nn.Sequential(*stage_blocks))
            dec_in_c = out_c
            cur_dp += dec_depths[i]

    def forward(self, x):
        # x: [B, C, H, W] -> 内部で最初のDownsamplingにより [B, H, W, C] になる
        
        # --- Encoder ---
        skips = []
        for i in range(self.num_enc_stages):
            x = self.downsamples[i](x)
            x = self.enc_stages[i](x)
            skips.append(x)
            
        # ボトルネック特徴 (Stage 5)
        out = skips.pop() 

        # --- Decoder ---
        for i in range(self.num_dec_stages):
            skip = skips.pop() # Stage 4, 3, 2, 1の順に取り出す
            out = self.upsamples[i](out, skip)
            out = self.dec_stages[i](out)
        
        # 一般的なセグメンテーションの出力形式 [B, C, H, W] に戻す
        out = out.permute(0, 3, 1, 2) 
        
        # 元の画像サイズに拡大
        # 最初のパッチ化で stride=4 なので、4倍にする
        out = F.interpolate(out, scale_factor=4.0, mode='bilinear', align_corners=False)
        
        return out
```

# 出力
## 機械学習と深層学習
機械学習(Machine Learning)と深層学習(Deep Learning)の違いについて、考えたことはあるだろうか。<br>
包含関係の話をすると、深層学習は、機械学習の一種とされている。<br>
つまり、機械学習のほうがより多くの要素を含む、広い概念である。<br>
一方で、両者の違いについて、実務的なところに注目して以下のように説明されることがある。

| 手法     | 説明                                                                                     |
| -------- | ---------------------------------------------------------------------------------------- |
| 機械学習 | 事前に定義された特徴量を使ってタスクを行う。学習可能なのは、**タスク実行の機構だけ**である。 |
| 深層学習 | 特徴量の抽出とタスクをどちらも行う。学習可能なのは、**特徴抽出とタスク実行の機構**である。   |

分かりにくいと思うので、具体例とともに説明しよう。

古典的な人物検出の手法として、HOG(Histograms of Oriented Gradients)[^8]が知られている。<br>
HOG特徴量は、さまざまな輝度勾配を検知するフィルタを事前に定義しておき、それを適用した結果を集約することで、人間らしさを表す。<br>
そして、HOG特徴量をSVM(Support Vector Machine)[^9]などの分類器に渡すことで、分類を行う。<br>
つまり、処理機構は`Input -> Backbone(HOG, not learnable) -> Classifier(SVM, learnable)`とまとめることができる。

一方で、最初の現代的なCNNとして紹介したAlexNet[^10]は、ILSVRC2012(ImageNet Large Scale Visual Recognition Challenge)という画像認識の世界大会において、従来の機械学習手法に大差をつけて優勝した。<br>
CNNの機構は人間が設計しているが、特徴抽出のためのパラメータはすべて機械が自動的に調整するという、革新的な手法で圧勝した。<br>
この処理機構は、分類器をClassifierとして`Input -> Backbone(CNN, learnable) -> Classifier(FCL, learnable)`とまとめることができる。

## FCL(Fully Connected Layer)
日本語では全結合層と呼ばれる機構で、大きさが`(..., C_in)`の特徴マップ`x`を入力すると、FCLは大きさ`(..., C_out)`の特徴マップ`y`を出力する。<br>
つまり、チャネル数を調整するような役割があるといえる。<br>
PyTorchでは`torch.nn.Linear`に実装されており、次のような計算を行う。

$$y=xA^T+b$$

`A`は`(C_out, C_in)`の行列、`b`は`(C_out)`のベクトルである。<br>
`b`はバイアス(Bias)と呼ばれるもので、`torch.nn.Linear`だけではなく、畳み込み層などのクラスで`True`、`False`を切りかえられる設計になっている。

`torch.nn.Linear`の計算式を単純化すると、一次関数$y=ax+b$と見なせる。<br>
$y=ax$だと原点を通る直線しか引けないが、バイアス`b`が加わることで、直線が`y`軸方向に移動できるようになり、表現力が向上する。<br>
そのため、特に意図がなければ、基本的にはバイアスは`True`で構わない。<br>
実際、PyTorchでは既定値で`bias=True`となっている。

以上の説明を踏まえて、U-Netの実装を振り返ると、興味深いことになっているのが分かる。<br>
`DoubleConv`の`nn.Conv2d`で`bias=False`に設定されているのだ。

```py:conv_layer.py
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

これには、`nn.BatchNorm2d`の性質が関係している。<br>
[計算式](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm2d.html)を見てみると、データの平均値$E[x]$が引かれて、スケールが調整された後、学習可能なバイアス$\beta$が足されている。<br>
つまり、`nn.Conv2d`で`bias=True`にしても、`nn.BatchNorm2d`で無意味になってしまう。<br>
そのため、`bias=False`とすることで無駄な計算を削減している。

## 出力層
### 分類モデル
[AlexNet](https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py)の出力層は、以下のように実装されている。

- ポイント1：`nn.AdaptiveAvgPool2d`によって、`self.avgpool`に入力される特徴マップの空間的な解像度を、指定されたもの（ここでは`(6, 6)`）に強制的に調整している。これは、次の`self.classifier`に入力する大きさを統一するための処理である。
- ポイント2：`nn.Dropout`によって、学習中にランダムで一部のニューロンが無効化される。これは、過学習を抑制するためのテクニックである。

```py:alexnet_classifier.py
import torch.nn as nn

# 以下、コメントは出力される特徴マップの大きさを表す
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 省略
        )                                             # (B, 256, height, width)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))   # (B, 256, 6, 6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),             # 9216 -> 4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),                    # 4096 -> 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),             # 4096 -> ターゲットのクラス数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # 空間とチャネルを押しつぶす (B, 256, 6, 6) -> (B, 256*6*6)
        x = self.classifier(x)    # (B, 256*6*6) -> (B, num_classes)
        return x
```

### セグメンテーションモデル

[U-Net](https://github.com/milesial/Pytorch-UNet)の出力層は、以下のように実装されている。

- ポイント1：`OutConv`の`nn.Conv2d`では`kernel_size=1`と設定されている。これは、`1*1`畳み込みと呼ばれる、近傍情報を一切見ない処理である。純粋に各ピクセルだけを見て、チャネル数を調整する役割がある。
- ポイント2：出力の変数が`logits`を名づけられている。これは、機械学習や深層学習で、モデルの生の出力を表すときに慣例的に使われる名前である。

```py:unet_classifier.py
import torch.nn as nn
from hoge import DoubleConv, Down, Up

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 簡略化してます
# コメントはチャネル数の推移
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels

        self.inc = (DoubleConv(n_channels, 64))
        # ...
        # ...
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))   # 64 -> n_classes

    def forward(self, x):
        x1 = self.inc(x)
        # ...
        # ...
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

# 損失関数
## 分類
### Cross Entropy
超王道の損失関数である。<br>
定義は[PyTorchのリファレンス](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)を参照されたい。<br>
感覚的な説明としては[この記事](https://qiita.com/kenta1984/items/59a9ef1788e6934fd962)が分かりやすい。

### Focal Loss
Cross Entropyを少し工夫した損失関数で、クラスの不均衡性に対処するために使われる。<br>
例えば、犬・猫・鹿の3クラス分類をするために、画像をインターネットで探したところ、犬と猫は1,000枚ずつ集まったのに、鹿は100枚しかなかったとする。<br>
この状態で学習を行うと、データ数が多い犬と猫に引っぱられて、鹿に対する推論性能が下がるという現象が起こりうる。<br>
このようなデータ数の不均衡を調整してくれるのが、Focal Lossである。<br>
医用画像処理の分野では、一般的に陽性の画像は陰性より圧倒的に少ないので、Focal Lossが有力な選択肢になる。

## セグメンテーション
### Dice Loss
セグメンテーションの評価指標の一つ、Dice係数を使った損失関数である。<br>
Dice係数は`[0, 1]`の値を取り、`1`に近いほど、モデルの推論性能が高いことを表す。<br>
それを逆手に取って、Dice Lossは`1-Dice`を損失として計算する。<br>
これにより、Dice係数を直接最適化することができる。

### Lovasz-softmax Loss[^11]
セグメンテーションの評価指標の一つ、IoU(Intersection over Union)の最適化を目的とした損失関数である。<br>
IoUの計算過程には、微分不可能な演算があるため、Dice係数と同じように`1-IoU`とすることはできない。<br>
Lovasz-softmax Lossでは、モデルの出力に対して変換をかけることで、IoUの最適化を実現している。

### Boundary DoU Loss[^12]
結局のところ、セグメンテーションで難しいのは、物体の境界部分である。<br>
そのため、境界部分での精度向上を目的とした損失関数も多数提案されている。<br>
その一つがBoundary DoU Lossで、予測と正解それぞれから輪郭だけを抽出して損失が計算される。

## その他
基本的な損失関数は上述のとおりだが、セグメンテーション用のものについては、[さらにいくつかまとめた記事](http://150.69.46.71:3000/survey/%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3/%E3%80%90JP%E3%83%BBENG%E3%80%91%E3%82%BB%E3%82%B0%E3%83%A1%E3%83%B3%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E7%94%A8%E3%81%AE%E6%90%8D%E5%A4%B1%E9%96%A2%E6%95%B0%E4%B8%80%E8%A6%A7)をGROWIに作成しているので、気になる人は確認すると良い。

## 使い方
Cross Entropyなどの代表的な損失関数は、`torch.nn`に実装されている。<br>
使用イメージは、以下のとおりである。

```py:loss_function.py
import torch.nn as nn

# 自作のクラスをインポート
from hoge import UNet, BasicDataset
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')

# 自作の損失関数をインポート
from hogehoge import dice_loss

def train_model(n_classes):
    # >>>>>>>>>> initialize data loader >>>>>>>>>>
    dataset = BasicDataset(dir_img, dir_mask)

    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # モデルを実体化する
    model = UNet(n_channels=3, n_classes=n_classes)

    # 損失関数を実体化する
    # 変数名として、基準という意味の criterion, criteria がよく使われる
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # データを読みこむためのループ
    # めっちゃ簡略化してます
    for batch in train_loader:
        # データを取得する
        images, true_masks = batch['image'], batch['mask']

        # モデルに渡して推論を行い、logitsを得る
        masks_pred = model(images)

        # 損失を計算する
        # セグメンテーションでは、ピクセルレベルの正確性を上げるCross Entropyと、
        # 領域ベースのDice Lossのように、性質の異なる損失関数を組み合わせるのが一般的
        if model.n_classes == 1:
            loss = criterion(masks_pred.squeeze(1), true_masks.float())
            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
        else:
            loss = criterion(masks_pred, true_masks)
            loss += dice_loss(
                F.softmax(masks_pred, dim=1).float(),# 確率マップへの変換
                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),# 正解データは 0 or 1 だけのマップに変換
                multiclass=True
            )
```
# train.pyを読んでみよう
例によって、プログラムは[U-Net](https://github.com/milesial/Pytorch-UNet)のものである。

## 1. ライブラリのインポート
| ライブラリ   | 役割                                                                                   |
| ------------ | -------------------------------------------------------------------------------------- |
| `argparse`   | `python3 train.py`のように実行するとき、コマンドラインから引数を受け取ることができる。 |
| `transforms` | データ拡張のライブラリである。                                                         |
| `optim`      | オプティマイザ(Optimizer)のライブラリである。                                          |
| `tqdm`       | `for`文の進行度を見やすく装飾する。                                                    |

```py:train_import.py
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from hoge import UNet
from hogehoge import BasicDataset, CarvanaDataset
from hogehogehoge import dice_loss

# パスの宣言
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
```

## 2. 学習関数の定義
次に、学習のための関数`train_model`が実装されている。<br>
変数名の意味は、あくまでこのプログラムにおけるものなので、実装によって異なることがある。

| 変数                | 意味                                                                                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`             | 実体化されたモデル。                                                                                                                                                                    |
| `device`            | GPUかCPUを指定する。                                                                                                                                                                    |
| `epochs`            | 学習の長さ。train用の画像が1,000枚あるとすると、その1,000枚を使いきるのが1エポックに相当する。                                                                                          |
| `batch_size`        | 一度にモデルに入力される画像の枚数。2の累乗や偶数が好んで使われる。                                                                                                                     |
| `learning_rate`     | 学習率。損失をモデルにフィードバックするときに、どれくらいパラメータを更新するかを表す。                                                                                                |
| `save_checkpoint`   | 特定のエポック数（1エポックなど）が終わるごとに、途中経過のモデル(Checkpoint)を保存するかのフラグ。                                                                                     |
| `amp`               | 自動混合精度を有効にするかのフラグ。速度重視の計算は16ビット、精度重視の計算は32ビットで行われる。有効にすると、GPUのVRAM使用量を削減でき、計算も速くなるが、精度が下がる可能性がある。 |
| `weight_decay`      | 重み減衰のパラメータ。過学習を抑制する役割がある。                                                                                                                                      |
| `momentum`          | パラメータ更新の度合いを表す。学習を高速化させる役割がある。                                                                                                                            |
| `gradient_clipping` | モデルにフィードバックする微分値（勾配, Gradient）の上限。勾配爆発を抑制するために設定される。                                                                                          |

```py:train_func.py
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # データセットを実体化
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # train/val に分割
    # ここでは val(= validation) と書かれているが、意味合い的には test と同義だと思われる
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # データローダを実体化
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # オプティマイザを実体化
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)

    # 学習率のスケジューラを実体化
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # 損失関数を実体化
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 学習ループ
    # エポック数の分だけ繰り返し処理を行う
    for epoch in range(1, epochs + 1):
        model.train()   # モデルを学習モードに切り替える
        epoch_loss = 0

        # for文の装飾
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # データを一回分読み込む処理を繰り返す
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                # データを device に転送
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 推論を実行
                    masks_pred = model(images)

                    # 損失を計算
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                
                # 最適化1：勾配をゼロで初期化
                optimizer.zero_grad(set_to_none=True)

                # 最適化2：損失をモデルにフィードバック
                # AMPの関係でscalerを経由
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)

                # 勾配爆発の対策
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                # 最適化3：オプティマイザも動かす
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 学習の途中経過を確認
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass
        # <<<<<<<<<< ここまでで1エポックが終わり <<<<<<<<<<

        # フラグが True なら、エポックごとにモデルを保存する
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
```

## 3. argparserの実装
コマンドラインから引数(Argument)を受け取れるようにすることで、実験条件を変えやすくする。<br>
`add_argument`で新しい引数を追加し、`parse_args()`で引数を辞書として取得する。

```py:train_args.py
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()
```

## 4. mainの処理
```py:train_main.py
if __name__ == '__main__':
    # 引数を受け取る
    args = get_args()

    # GPUが使用可能かどうか判定して、device という変数に保存
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # ↑ 元のコメントそのまま
    #   n_channels と n_classes は、この分野の人であればコメントがなくても
    #   意味は分かるはず。このプログラムの著者は親切だ
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    
    # 学習済みモデルを読み込む
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    
    # model を device に転送
    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    
    # OOM(Out of Memory)の処理
    # OOMは、GPUのVRAM不足を意味する
    # モデルの設定（チャネル数、深さなど）、バッチサイズ、画像の大きさなどを見直そう
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
```

# 必須テクニック
## 新しいモデル・データセット
はじめて深層学習をするときは、既存のプログラムを流用することが多いだろう。<br>
しかし、理解が深まるにつれて、新しいモデルやデータセットを使いたくなる。<br>
次のようなことでつまづくかもしれないが、データの入力方法、すなわちデータセットクラスさえ調整できれば、解決できることが多い。

- 別のプロジェクトに実装されたモデルを試したいけど、今まで使っていたデータセットのクラスがないな……。<br>➡プログラムの著者が変わると、`train.py`やデータセットの要件も変わる。基本的な概念は共通しているが、細かい仕様は千差万別だ。新しいプロジェクトのデータセットクラスを見て、今までのデータセットクラスを書きかえよう。
- 新しいデータセットを試したいけど、やり方が分からないな……。<br>➡このパターンのほうが、問題はいくらか単純である。作業するプロジェクトは同じなので、既存のクラスを参考にして、同じものを`return`するように、新しいデータセットクラスを実装すれば良い。そして、それが実体化されるように、設定ファイルや`train.py`など、必要な個所を見つけて書きかえる。

### 具体例
入力の章で紹介したデータセットクラスを再掲する。

```py:dataset.py
from pathlib import Path
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        # ディレクトリの中にあるファイルを検索して、self.idsにリストとして保存

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # 画像の前処理
        if is_mask:
            # mask(セグメンテーションの正解データ)の処理
            return mask

        else:
            # 画像の処理
            return img

    def __getitem__(self, idx):
        name = self.ids[idx]   # インデックスに対応するファイル名を取得

        # imgに画像を読み込んでpreprocessで処理
        # maskに正解データを読み込んでpreprocessで処理

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
```

このクラスでは、`__getitem__`の中で、画像と正解データを別々に読み込んでいる。<br>
しかし、今から使おうとしているデータセットでは、どちらも`.npy`ファイルに辞書としてまとめて保存されているとする。<br>
これでは、`BasicDataset`をそのまま使えないので、新しいクラスを実装することになる。<br>
この場合は、`Dataset`を継承して最初から実装するのではなく、`BasicDataset`を継承して、必要な部分だけオーバーライドするのが賢い方法だ。

- `__init__`：`.npy`にまとめられているため、受け取るディレクトリは一つで十分。
- `__getitem__`：個別に画像を読み込んでいた処理を`np.load`にする。

```py:new_dataset.py
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch

class BasicDataset(Dataset):
    # 省略

# 新しいクラスを実装
class NpyDataset(BasicDataset):
    def __init__(self, npy_dir: str):
        self.npy_dir = Path(npy_dir)
        self.ids = [p.stem for p in self.npy_dir.glob('*.npy')]   # npyファイルを検索して self.ids にリスト化

    def __getitem__(self, idx):
        name = self.ids[idx]
        npy_path = self.npy_dir / f"{name}.npy"

        # npyファイルを読み込む
        data = np.load(npy_path, allow_pickle=True).item()
        img = data['image']
        mask = data['mask']

        # imgとmaskをpreprocessで処理

        # 同じ形式でreturn
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
```

## 転移学習とファイニューニング
### データ拡張の限界
かぎられた資源をフル活用するための技術として、データ拡張について紹介した。<br>
より高度なデータ拡張手法もさまざまあるが、それでも、疑似的な拡張には限界がある。<br>
やはり、元のデータ数が多いほうが、汎化性能の向上には望ましい。

そこで登場するのが、転移学習(Transfer Learning)、ファインチューニング(Fine-tuning)と呼ばれる技術である。<br>
この二つはよく似た概念で、「データセットAでの学習結果を初期パラメータとして、データセットBで学習を始める」という部分が共通している。<br>
逆に、以下の部分が異なる。

- 転移学習：モデルのうち、出力層のみをLearnableとして、それ以外はFreezeさせる（＝パラメータが更新されない）。主要な特徴抽出能力は元のままで、Classifierだけが調整される。
- ファインチューニング：より広い範囲（出力側の数層、あるいはモデル全体）をLearnableとする。一般的には、転移学習より高精度となりやすい。

文脈から判断して、転移学習・ファインチューニングを使う場面では、データセットの規模は「A > B」となる。<br>
AとBの画像は、似ているほうが望ましいが、まったく異なるものを使ったとしても、効果があることも多い。

例えば、データセットAとして犬の画像（柴犬、プードル、チワワなど）を集めて、品種分類のモデルを学習させたとする。<br>
次に、データセットBとして猫の画像（ラグドール、アメリカンショートヘア、マンチカンなど）を集める。<br>
犬と猫は、どちらも四足歩行で、耳と尻尾があって、毛で覆われていて、可愛いなど、共通点がたくさんある。<br>
そのため、犬の品種分類で学んだ知識は、猫の品種分類でも役に立つだろうと考えられる。

一方で、データセットBとして車の画像（トヨタ・ハイエース、日産・セレナ、ダイハツ・ムーヴなど）を集める。<br>
犬と車では、見た目がまったく異なる。<br>
しかし、深層学習モデルは、犬の品種分類を通して、エッジへの応答、大域的な関連性の理解など、人間の直感では分からない特徴抽出の能力を獲得している。<br>
この汎用的な知識は、画像の種類に関係なく有用なので、ファインチューニングの効果がある程度は期待できる。

### 具体例
[U-Net](https://github.com/milesial/Pytorch-UNet)をベースに、Geminiと相談のうえ、木原が書いてみた。<br>
以下の例では、データセットAとBの中身が違うことを想定している。<br>
両者の中身が同じ場合（e.g., 同じ車の画像で、Pre-trainingはネット上の大規模データセット、Fine-tuningは自作の小規模データセットを使う）などは、`replace_classifier`が必要なく、よりシンプルに実装できる。

また、プログラムの中で`backbone`という言葉を使っている。<br>
これは、`logits`を出力する`model`だと思っておくと良い。<br>
つまり、`model`と`backbone`は同義である。<br>
著者によっては、`self.backbone = build_backbone(config)`、`self.classifier = build_classifier(config)`として、特徴抽出器と分類器を明確に切り分けている、汎用性の高い実装がされていることもある。

```py:fine_tuning.py
import torch
import torch.nn as nn
from hoge import UNet

def load_pretrained_model(
    model_path,
    n_channels,
    n_classes,
    device,
    freeze_backbone=False,
    replace_classifier=False
):
    # モデルを普通に実体化
    model = UNet(n_channels=n_channels, n_classes=n_classes)

    # 学習済みモデルを読み込む
    state_dict = torch.load(model_path, map_location=device)

    # データセットが変わると、出力層の意味合いが変わるため、削除
    # 状況によっては、そのまま残しても良い
    if replace_classifier:
        # このU-Netの場合、最後の層の名前は 'outc.conv.weight' と 'outc.conv.bias'
        del state_dict['outc.conv.weight']
        del state_dict['outc.conv.bias']

    # 学習済みモデルでパラメータを置きかえる
    # strict=False にしておくと、パラメータ名が一致するものだけに適用
    # 不要なパラメータがあれば、keyを指定することで削除可能
    # del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    # freezeの設定
    if freeze_backbone:
        for name, param in model.named_parameters():
            # 出力層 outc 以外を更新不可にする
            if 'outc' not in name:
                param.requires_grad = False
    
    # model = UNet() でランダム初期化済み
    # model.load_state_dict() で学習済みモデルを読み込んだが、
    # outcは上書きされていない
    # つまり、outcはランダム初期化のままなので、これでreturnしてOK
    return model
```

## モデルのカスタマイズ
深層学習モデルの特徴抽出をサポートする部品は、一般的にモジュール(Module)と呼ばれる。<br>
目的に応じて、モジュールはモデルの各所に配置される。

### CBAM(Convolutional Block Attention Module)
CBAM[^13]は、最も有名なAttentionモジュールの一つである。<br>
Channel Attention ModuleとSpatial Attention Moduleを直列につなげた構造だ。

ちなみに、Attentionは「注意」という意味で、特徴マップの中で注目するべきところを強調するのがAttentionモジュールの役割である。<br>
一般的に、注目というのは、`[0, 1]`の値を持ったアテンションマップを特徴マップにかけ算することを指す。<br>
これにより、`0`に近いほど特徴は抑制され、`1`に近いほど相対的に強調されることになる。

![cbam.png](/attachment/6a040942e4ba3f4e1317e0b0)

- Channel Attention Module：チャネル数を`C`とすると、`(C)`のアテンションマップを作成して、空間方向に適用することで、チャネルの強弱を調整する。
- Spatial Attention Module：特徴マップの空間的な解像度と同じ大きさのアテンションマップを作成して、チャネル方向に適用することで、空間の強弱を調整する。

実際に[CBAMの実装](https://github.com/luuuyi/CBAM.PyTorch)を見てみよう。

```py:cbam.py
import torch.nn as nn

# 3x3畳み込みを簡単に導入するための関数
# planes という単語が、チャネル数(channels)を意味することもよくある
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Fully Connected Layer [conv -> ReLU -> conv]
        # 隠れ層のチャネル数を 1/16 に圧縮
        # ratio で圧縮率を指定する意図だと思われるが、マジックナンバになってしまっている
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        
        # 値を[0, 1]にスケーリングするので、アテンションマップを作成する目的で使われることが多い
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均プーリングで空間情報をつぶす
        avg_out = self.fc(self.avg_pool(x))   # (B, C, 1, 1)

        # 最大値プーリングで空間情報をつぶす
        max_out = self.fc(self.max_pool(x))   # (B, C, 1, 1)

        # 二つの方法で得られた情報を足す
        out = avg_out + max_out   # (B, C, 1, 1)

        # sigmoidに通してアテンションマップを作成
        return self.sigmoid(out)   # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        # プーリングの結果が入力されるので、in_channels=2
        # 空間方向のアテンションマップを作成するので、out_channels=1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

        # スケーリングのsigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # dim=1 で空間方向にプーリング
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B, 1, H, W)

        # torch.cat でチャネル方向にプーリング結果を統合
        x = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)

        # conv でチャネル数を調整
        x = self.conv1(x)   # (B, 1, H, W)

        # sigmoidに通してアテンションマップを作成
        return self.sigmoid(x)   # (B, 1, H, W)

# 明示的に CBAM というクラスを作ることもあるが、
# この実装では、BasicBlock の中で直接二つのAttentionが適用されている
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        # 畳み込み層などを宣言
        # 直列での処理なので、nn.Sequential にまとめても良い
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # CBAM の構成要素を宣言
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差接続用
        residual = x

        # conv -> batch norm -> relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv -> batch norm
        out = self.conv2(out)
        out = self.bn2(out)

        # CBAM の処理
        # 上記の図と同じ処理が行われていることが分かる
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差接続
        out += residual

        # 最後に活性化関数を通す
        out = self.relu(out)
        return out
```

### 拡張畳み込み(Dilated Convolution, or Atrous Convolution)
受容野のところで説明したように、`nn.Conv2d`のパラメータ`kernel_size`を大きくすると、受容野は広くなるが、計算量が爆発的に増えるのが懸念点だった。<br>
そこで、計算量を抑えつつ受容野を広げる方法として、拡張畳み込み[^14]が知られている。<br>
[このページのアニメーション](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)が分かりやすいので、参考にすると良い。

拡張畳み込みの実装には、特別な工夫は不要である。<br>
`nn.Conv2d`に`dilation`というパラメータがあるからだ。<br>
既定値`dilation=1`が通常の畳み込みで、これを大きくすると拡張畳み込みになる。<br>
実際に、これまでに紹介してきたU-Netの`DoubleConv`を、拡張畳み込み仕様に変えてみよう。

```py:dilated_conv.py
import torch.nn as nn

class DilatedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # dilationとpaddingを指定
        # paddingもdilationに合わせておくことで、特徴マップの空間的な解像度が変わるのを防ぐ
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

あとは、これをU-Netの畳み込み層として`__init__`で実体化するだけだ。<br>
すべてを拡張畳み込みにするのではなく、ダウンサンプリングが進んだ深い層の一部を`DilatedDoubleConv`に置き換え、受容野をさらに広げるのが、効果が期待できる実装方法だと思われる。<br>
~~たぶんね。~~

# 最後に
間違いを見つけたり、内容をアップデートしたりするときは、何も言わずに記事を編集しても構わない。<br>
それがGROWIの良いところである。<br>
ただ、どこが変わったのか分かるように、コメントを残すなどしてもらえると、ありがたい。

[^1]: K. Fukushima, “Neocognitron: A Hierarchical Neural Network Capable of Visual Pattern Recognition”, Neural Networks, Vol. 1, pp. 119-130, 1988.
[^2]: Y. Lecun et al., “Gradient-based Learning Applied to Document Recognition”, Proceedings of the IEEE, Vol. 86, No. 11, pp. 2278-2324, 1998.
[^3]: K. He et al., “Deep Residual Learning for Image Recognition”, Computer Vision and Pattern Recognition, pp. 770-778, 2016.
[^4]: Y. Wu et al., "Group Normalization", European Conference on Computer Vision, pp. 3-19, 2018.
[^5]: A. Vaswani et al., “Attention Is All You Need”, International Conference on Neural Information Processing Systems, pp. 6000-6010, 2017.
[^6]: J. Alammar, "The Illustrated Transformer", 2018, https://jalammar.github.io/illustrated-transformer/, Accessed 2026/4/26.
[^7]: A. Dosovitskiy et al., "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale", International Conference on Learning Representations, 2021.
[^8]: N. Dalal et al., "Histograms of Oriented Gradients for Human Detection", Computer Vision and Pattern Recognition, pp. 886-893, 2005.
[^9]: C. Cortes et al., "Support-vector Networks", Machine Learning, Vol. 20, pp. 273–297, 1995.
[^10]: A. Krizhevsky et al., “ImageNet Classification with Deep Convolutional Neural Networks”, Advances in Neural Information Processing Systems, pp. 1106-1114, 2012.
[^11]: M. Berman et al., "The Lovász-softmax loss: A Tractable Surrogate for the Optimization of the Intersection-over-union Measure in Neural Networks", Computer Vision and Pattern Recognition, pp. 4413-4421, 2018.
[^12]: F. Sun et al., "Boundary Difference over Union Loss for Medical Image Segmentation", Medical Image Computing and Computer Assisted Intervention, pp. 292–301, 2023.
[^13]: S. Woo et al., "CBAM: Convolutional Block Attention Module", European Conference on Computer Vision, pp. 3-19, 2018.
[^14]: F. Yu et al., "Multi-Scale Context Aggregation by Dilated Convolutions", International Conference on Learning Representations, 2016.