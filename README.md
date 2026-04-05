# NBD-Dirichlet マーケットシェアモデル

Goodhardt, Ehrenberg & Chatfield (1984) の NBD-Dirichlet モデルを Julia + Turing.jl で再現したノートブック。

## 参考文献

Goodhardt, G.J., Ehrenberg, A.S.C. & Chatfield, C. (1984).  
"The Dirichlet: A Comprehensive Model of Buying Behaviour."  
*Journal of the Royal Statistical Society*, Series A, **147**(5): 621–655.

## モデル概要

| 段階 | 問い | 分布 |
|------|------|------|
| カテゴリ購買 | 何回買うか？ | 負の二項分布 (NBD) |
| ブランド選択 | どのブランドを選ぶか？ | Dirichlet-Multinomial |

**パラメータ**（UK インスタントコーヒー市場、24週間）

| パラメータ | 値 | 意味 |
|------------|-----|------|
| M | 4.3 | カテゴリ平均購買回数 |
| K | 0.78 | NBD 集約パラメータ |
| S | 1.26 | Dirichlet 総集約パラメータ |

## ファイル構成

```
.
├── Dirichlet_MarketShare_Turing.ipynb   # メインノートブック
├── NBD_obsv_ideal.png                   # NBD 理論値 vs 観測値
├── param_real_infer.png                 # パラメータ事後分布
├── real_infer_observ.png                # 浸透率・購買頻度の比較
├── double_jeopardy.png                  # Double Jeopardy 可視化
├── README.md
└── CLAUDE.md
```

## ノートブック構成

1. **理論的背景** — NBD・Dirichlet-Multinomial の数式
2. **論文データの設定** — UK コーヒー市場パラメータ
3. **理論値の計算** — 浸透率・購買頻度を数値積分で算出
4. **シミュレーションデータ生成** — 真パラメータから個人レベルデータを生成
5. **Turing.jl によるベイズ推定** — NUTS サンプラーで MCMC
6. **収束診断** — R̂、トレースプロット、事後分布プロット
7. **パラメータ推定結果** — 真値との比較
8. **主要指標の予測と比較** — 浸透率・購買頻度
9. **Double Jeopardy 法則** — シェア vs 浸透率/購買頻度の可視化

## 環境構築

```julia
import Pkg
Pkg.add(["Turing", "Distributions", "StatsPlots", "DataFrames",
         "SpecialFunctions", "MCMCChains", "StatsBase"])
```

**Julia バージョン**: 1.10 以上推奨

## 実行手順

1. Jupyter Lab / Jupyter Notebook を起動
2. `Dirichlet_MarketShare_Turing.ipynb` を開く
3. セルを上から順に実行（Cell 3 のパッケージインストールは初回のみ）
4. MCMC サンプリング（Cell 18）は数分かかる

## 主要な出力指標

- **浸透率** $b_j$：期間中に少なくとも1回購買した消費者の割合
- **平均購買頻度** $w_j$：購買者1人あたりの平均購買回数
- **Double Jeopardy**：小シェアブランドは浸透率・購買頻度とも低い

## 注意事項

- MCMCChains からのパラメータ取得は `get(chain, :M).M` および `namesingroup(chain, :p)` を使用
- `Array(chain)` は環境によって 2D / 3D の返り値が異なるため使用しない
- ノートブックを更新後は Jupyter で「File → Revert Notebook」で再読み込みが必要
