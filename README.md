这是ASVSpoof 2019 & 2021的一个简单实现样例

## 文件功能

| 文件名       | 功能说明                 |
| ------------ | ------------------------ |
| config.py    | 参数配置                 |
| dataset.py   | 从硬盘加载数据           |
| frontend.py  | 从语音波形中提取声学特征 |
| metrics21.py | 评估指标                 |
| model.py     | 神经网络                 |
| train.py     | 训练代码，包含预测代码   |

## 依赖

```
numpy
librosa
pandas
torch
torchaudio
```

## 使用说明

1.修改参数配置，尤其是`data_root`，目录需遵循如下格式：

```bash
 data_root/
	LA2019/
		ASVspoof2019_LA_asv_protocols/
			...
		ASVspoof2019_LA_asv_scores/
			...
		ASVspoof2019_LA_cm_protocols/
			ASVspoof2019.LA.cm.dev.trl.txt
			ASVspoof2019.LA.cm.train.trn.txt
			...
		ASVspoof2019_LA_dev/
			flac/
			...
		ASVspoof2019_LA_eval/
			...
		ASVspoof2019_LA_train/
			flac/
			...
		README.LA.txt
	LA2021/
		ASVspoof2021_LA_eval/
			flac/
			ASVspoof2021.LA.cm.eval.trl.txt
			...
		keys/
			ASV/
			CM/
	DF2021/
		ASVspoof2021_DF_eval/
			flac/
			ASVspoof2021.DF.cm.eval.trl.txt
			...
		keys/
			CM/
```

2.搭建神经网络模型，本项目提供了样例模型

3.开始训练并测试

```
python train.py
```

