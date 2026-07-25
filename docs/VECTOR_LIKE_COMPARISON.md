# SVG的構造潜在による x4 比較

> 抽出元Upscalerリポジトリで行った履歴評価である。記載された`Images/`と
> `outputs/`のパスおよびHAT比較は抽出元を参照し、SDF-Upscaler配布物には含めない。

## 比較の定義

ここでいう「SVG的中間表現」は SVG ファイル、パスへの変換、SVG
レンダラーのいずれでもない。`LLM.md` で提案した、輪郭を局所的な幾何として扱う
**vector-like structural latent** を指す。

- RGBA: alpha の 0.5 等値線から Segment Graph を作り、Segment-SDF で被覆率を再構成する。
- RGB: マルチスケールエッジを開輪郭のままスケルトン化し、局所 SDF/UDF、法線、接線、
  曲率、接続点、両側色コントラストを密な構造フィールドとして持つ。
- RGB の今回の試験では、構造フィールドから作る線形光 RGB 残差を HAT 出力へ加え、
  LR 再投影で改善する位置だけを採用する。これは `LLM.md` の Stage 1 output gate に相当する。

## 対象と条件

| 経路 | 入力 | ベースライン | 構造潜在あり |
|---|---|---|---|
| RGBA alpha | `Images/image.png` (1040x1675) | Lanczos x4 | calibrated Segment-SDF x4 |
| RGB | `Images/image2.png` (1920x1080) | HAT classical x4 | HAT + edge-local structure residual + LR gate |

どちらも独立した HR 正解画像を持たない単画像試験である。LR 再投影誤差は入力との
自己整合性を測るが、知覚品質やベンチマーク性能の代用にはならない。

## 結果

### RGBA alpha

| 指標 | 結果 |
|---|---:|
| 1 LSB 以上変化した HR alpha | 0.8025% |
| 全画素 alpha MAE | 0.000773 |
| LR の partial-alpha 領域における alpha MAE | 0.022532 |
| LR 再投影 MAE: Lanczos | 0.00090016 |
| LR 再投影 MAE: Segment-SDF | 0.00074130 |
| LR 再投影 MAE の相対低下 | 17.65% |

変化は髪、羽、指、衣装などの細い silhouette 帯へ集中した。差分では正負が輪郭の
両側に対で現れ、画像内部のテクスチャ変更ではなく、輪郭位置と subpixel coverage
の調整になっている。今回の2候補では Segment-SDF の方が入力 alpha への再投影も良い。

[RGBA 拡大比較](../outputs/vector_like_compare/image_rgba/comparison_crops.png) /<br>
[全体 alpha 差分 x8](../outputs/vector_like_compare/image_rgba/alpha_delta_8x.png) /<br>
[HAT RGB + Lanczos alpha](../outputs/vector_like_compare/image_rgba/hat_lanczos_alpha_rgba_4x.png) /<br>
[HAT RGB + Segment-SDF alpha](../outputs/vector_like_compare/image_rgba/hat_segment_sdf_rgba_4x.png) /<br>
[数値](../outputs/vector_like_compare/image_rgba/metrics.json)

### RGB

| 指標 | LR gate あり | LR gate なし |
|---|---:|---:|
| 1 LSB 以上変化した HR pixel | 0.4191% | 6.9689% |
| HAT からの RGB MAE (8-bit) | 0.01136 | 0.24675 |
| LR 再投影 MAE | 0.00076900 | 0.00098585 |

HAT 単独の LR 再投影 MAE は 0.00077624 で、gate ありでは 0.93% 低下した。
変更はランプ枠、テーブルの境界、細い建築線などへ限定された。一方、gate なしでは
変化量は大きくなるが、再投影誤差は HAT 単独より悪化した。したがって、RGB では構造潜在
そのものだけでなく、confidence/support/reprojection gate が必須である。

[RGB 拡大比較](../outputs/vector_like_compare/image_rgb/comparison_crops.png) /<br>
[全体 RGB 差分 x8](../outputs/vector_like_compare/image_rgb/hat_structure_delta_8x.png) /<br>
[数値](../outputs/vector_like_compare/image_rgb/metrics.json)

### RGB feature fusion (M1 learned FiLM)

The next experiment moved the same 11-channel vector-like latent inside HAT.
HAT stayed frozen and zero-initialized FiLM heads after all six RHAG groups were
trained for 2,000 steps on the 800 official DIV2K x4 pairs.

On 100 fixed DIV2K validation crops, conditioned HAT improved from
`31.45460 dB` to `31.46670 dB` (`+0.01211 dB`).  On `image2.png`, modifications
were 25.5 times stronger at high-confidence structural locations than at flat
locations, while LR reprojection error improved by 0.86% relative to HAT.

[M1 learned-FiLM crops](../outputs/vector_like_compare/image_rgb/hat_film_div2k_2k_comparison_crops.png) /<br>
[M1 x4 output](../outputs/vector_like_compare/image_rgb/hat_film_div2k_2k_4x.png) /<br>
[M1 metrics](../outputs/vector_like_compare/image_rgb/hat_film_div2k_2k_metrics.json)

## 判定

SVG的構造潜在を使うことで、両経路とも出力は変化した。ただし意味は異なる。

- RGBA alpha は、古典的な明示幾何だけでも有用性が比較的強く、輪郭限定の変更と
  再投影改善が同時に得られた。
- RGB の出力後残差は安全な proof of concept だが、ゲート後の効果は小さい。
  HAT が既に作った輪郭位置を後段だけで直す限界が見える。
- zero-init FiLM による特徴空間融合は実際に学習可能で、最初の全DIV2K runでは
  HAT単独に対する小さい正のvalidation差を確認した。
- 次はM1の長期学習と edge-map-only ablationを行い、Urban100などの独立HR正解で
  PSNR/SSIM、輪郭位置、自然画像への非劣化を検証する。現時点ではSOTA主張ではない。
