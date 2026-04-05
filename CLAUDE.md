# CLAUDE.md — NBD-Dirichlet プロジェクト向けガイダンス

## プロジェクト概要

Julia + Turing.jl による NBD-Dirichlet マーケットシェアモデルの実装と検証。
ノートブック: `Dirichlet_MarketShare_Turing.ipynb`

## 重要な既知事項

### MCMCChains API の正しい使い方

`Array(chain)` は MCMCChains のバージョンによって 2D / 3D の返り値が異なり不安定。
**必ず以下の公式 API を使うこと**：

```julia
# スカラーパラメータの事後平均
M_est = mean(get(chain, :M).M)
K_est = mean(get(chain, :K).K)
S_est = mean(get(chain, :S).S)

# ベクトルパラメータ（Dirichlet p）の事後平均
p_syms = namesingroup(chain, :p)   # [Symbol("p[1]"), Symbol("p[2]"), ...]
p_vals = get(chain, p_syms)        # NamedTuple
p_est  = [mean(p_vals[s]) for s in p_syms]
```

**使ってはいけないパターン**:
```julia
# NG: バージョンによって 2D/3D が変わる
n_iter, n_params, n_chains = size(Array(chain))

# NG: PrettyTables の tf_unicode_rounded / crayon は環境依存
pretty_table(df; header_crayon=crayon"bold", tf=tf_unicode_rounded)
```

### テーブル表示

`PrettyTables.jl` は使わず `display(df)` を使う。Jupyter 上では DataFrames が HTML テーブルとして描画される。

### 変数名

`α_est` は Julia の Unicode 変数名。bash の `-c "..."` でスクリプトを書く際に `$` や Unicode 変数が展開・消去されることがある。Python スクリプトはファイルとして書いてから実行すること。

## ノートブック編集の注意

Jupyter はブラウザ上にスナップショットを保持する。ディスク上のファイルを外部から変更しても自動反映されない。セルを変更した場合はユーザーに「当該セルを手動で書き換えるか、File → Revert Notebook で再読み込み」を案内する。

## コード生成ルール

- Julia パッケージのインストールは `Pkg.add([...])` を使う（`]` REPLモードは Jupyter 非対応）
- スカラー出力は `println(@sprintf(...))` を使う
- プロット保存は `savefig("filename.png")` を直後のセルに分離する
- MCMC は `MCMCSerial()` をデフォルトとし、マルチスレッド版はコメントアウトで併記する

## 環境情報

- **言語**: Julia 1.10+
- **主要パッケージ**: Turing, Distributions, StatsPlots, DataFrames, MCMCChains, SpecialFunctions
- **サンプラー**: NUTS(500, 0.65)、1000 サンプル × 2 チェーン
- **データ**: UK インスタントコーヒー市場（Goodhardt et al. 1984）、N=600 シミュレーション消費者
