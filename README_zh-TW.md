# 韌性材料供應鏈合成基準測試

本專案提供 3,255 筆合成材料申請、84 份虛構政策，以及分類、檢索、消融、證據缺漏、成本與談判模擬程式。

## 執行方式

使用 Python 3.10 或更新版本，在虛擬環境中執行：

```bash
python -m pip install -r requirements-lock.txt
python run_benchmark.py --output artifacts_new_run
python -m unittest discover -s tests -v
```

## 資料與結果

資料位於 artifacts/data，結果位於 artifacts/results。請先閱讀 artifacts/results/REPORT.md 與 docs/PROVENANCE.md。此實作依論文規格產生新的合成紀錄，部分輸出與稿件數值不同，差異見 artifacts/results/paper_comparison.csv。引用時應標明實際使用的 commit，並依對應輸出報告結果。

本專案使用虛構政策與合成資料；成本與談判延遲屬數值模型。

## 公開與授權

公開網址：https://github.com/wltsai88/Resilient-Material-Supply-Chains

自行建立的資料、結果與文件採 CC BY 4.0，詳見 DATA_LICENSE.md；程式碼與可執行腳本採 MIT，詳見 LICENSE。第三方相依套件保留各自授權。引用資訊見 CITATION.cff。
