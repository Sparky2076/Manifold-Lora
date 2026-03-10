## 2026-03-10 DistilBERT + LoRA on SST2 (job 307845)

- 模型: distilbert-base-uncased
- 数据集: GLUE SST2, text_field = sentence
- 训练参数: epochs=50, batch_size=4, grad_accum_steps=8, lr=1e-5, max_length=128
- LoRA: type=default (lora.py), r=8, alpha=16, dropout=0.05, attention_only=True
- 运行脚本: scripts/run_train_bsub.sh + scripts/submit_bsub.sh 默认配置

文件说明:

- train.csv: iteration,train_loss,train_accuracy
- test.csv:  iteration,test_loss,test_accuracy

