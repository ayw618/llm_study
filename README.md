# LLM 学习仓库
这是一个用于本人学习大模型的仓库

Windows系统 PowerShell 
```powershell
python -m venv .llm_study_venv
.llm_study_venv\Scripts\activate
```
## Post-Training

### GRPO
文档在 [post_training/grpo/docs/GRPO.md](post_training/grpo/docs/GRPO.md)
环境配置：
```powershell
pip install -r post_training/grpo/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
运行命令：
```powershell
python post_training\grpo\simple_dum_grpo.py
```
### DAPO
文档在 [post_training/dapo/docs/DAPO.md](post_training/dapo/docs/DAPO.md)
环境配置同GRPO
```powershell
pip install -r post_training/grpo/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行命令：
```powershell
python post_training\dapo\simple_dum_dapo.py
```

### OPD
文档在 [post_training/opd/docs/OPD.md](post_training/opd/docs/OPD.md)
环境配置同GRPO
```powershell
pip install -r post_training/grpo/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行命令：
```powershell
python post_training\opd\simple_opd.py
```