```text
如果自己特定領域的資料 (像是醫療)，是可以對 Amazon Bedrock 裡提供的大模型做 fine tuning 嗎? A: 因為現在的大模型多是通用型，所以對於某些更為專業的領域，是可以透過進階的微調做精進 大約的作法是準備好 (數百或數千則) 該領域的資料，上傳到 Amazon S3，然後到 Bedrock 選擇已經開放微調的大模型，例如 Claude 3 Haiku，即可交由 Bedrock 做進階的訓練 您可以參考 [1], [2], [3] [1] https://aws.amazon.com/blogs/aws/fine-tuning-for-anthropics-claude-3-haiku-model-in-amazon-bedrock-is-now-generally-available/ [2] https://community.aws/content/2jNtByVshH7vnT20HEdPuMArTJL/mastering-amazon-bedrock-custom-models-fine-tuning-part-1-getting-started-with-fine-tuning [3] https://community.aws/content/2m9HnIvqSQ5Y3Jo03Nz48PlKpT5/mastering-amazon-bedrock-custom-models-fine-tuning-part-2-data-preparation-for-fine-tuning-claude-3-haiku





Q:
想請問 公司有專門領域 在醫療相關的資料 這樣 建議應該走哪一個方案呢?
A:
Q: 如果自己特定領域的資料 (像是醫療)，是可以對 Amazon Bedrock 裡提供的大模型做 fine tuning 嗎? A: 因為現在的大模型多是通用型，所以對於某些更為專業的領域，是可以透過進階的微調做精進 大約的作法是準備好 (數百或數千則) 該領域的資料，上傳到 Amazon S3，然後到 Bedrock 選擇已經開放微調的大模型，例如 Claude 3 Haiku，即可交由 Bedrock 做進階的訓練 您可以參考 [1], [2], [3] [1] https://aws.amazon.com/blogs/aws/fine-tuning-for-anthropics-claude-3-haiku-model-in-amazon-bedrock-is-now-generally-available/ [2] https://community.aws/content/2jNtByVshH7vnT20HEdPuMArTJL/mastering-amazon-bedrock-custom-models-fine-tuning-part-1-getting-started-with-fine-tuning [3] https://community.aws/content/2m9HnIvqSQ5Y3Jo03Nz48PlKpT5/mastering-amazon-bedrock-custom-models-fine-tuning-part-2-data-preparation-for-fine-tuning-claude-3-haiku

```

請問在處理 LLM App 上，使用 EC2 GPU 跟 Bedrock Service 這兩個差異你們有比較詳細的比較資料可以參考嗎？ 例如什麼樣的場景適合走 Bedrock，什麼樣的場景可能在成本效益上更適合用 EC2 ? （假設不需要考量維運問題的狀況）
A:
Amazon Bedrock 可以比擬成一個 SaaS 的服務，裡面有 Knowledge Base [1]、提示詞代管 [2]、不當內容管控 [3]、模型間評比 [4]、工作代理 [5] 等開箱即用的功能，您也不必擔心底層的維運或擴展 Amazon SageMaker 可以視作工作平臺，您可以大規模構建、訓練和部署機器學習模型。這包括使用筆記本、調試器、分析器、管道和MLOps等工具從頭開始構建大模型。另外，Amazon Bedrock 目前僅托管了約 50 個左右的大模型，但 SageMaker 提供對數百個預訓練模型的訪問，包括公開可用的 Hugging Face 上的大模型 更多合適的使用場景，請參考 [6] [1] https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html [2] https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html [3] https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html [4] https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation.html [5] https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html [6] https://docs.aws.amazon.com/decision-guides/latest/bedrock-or-sagemaker/bedrock-or-sagemaker.html



請問資料庫遷移可以將本地的MySQL遷移到AWS上的oracle嗎?
A:
如果您想要將地端的 MySQL 資料庫遷移到 Amazon RDS for Oracle [1]，是可以透過 AWS 提供的全托管工具 DMS [2]，再配合欄位型態的轉換工具 SCT [3] 來完成該遷移工作 但要提醒您的是，這兩種資料庫 (MySQL v.s. Oracle) 的欄位定義與引擎行為模式有比較大的差異，請務必在非生產環境做完詳細的語法檢查與壓力測試，再進行生產環境的遷移 [1] https://aws.amazon.com/rds/oracle/ [2] https://aws.amazon.com/dms/features/ [3] https://docs.aws.amazon.com/SchemaConversionTool/latest/userguide/CHAP_Source.html