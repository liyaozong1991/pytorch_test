# 数据格式
输入文件格式为json或[jsonl](https://jsonlines.org/)文件
### SFT
json
```json
[ {"prompt":"xxx", "output":"xxx"},
  {"prompt":"xxx", "output":"xxx"} ]
```
jsonl
```json lines
{"prompt":"xxx", "output":"xxx"}
{"prompt":"xxx", "output":"xxx"}
```

### RM/DPO
```json lines
{"prompt":"xxx", "chosen":"xxx", "reject":"xxx"}
```

### RLHF
jsonl
```json lines
{"prompt":"xxx"}
```
