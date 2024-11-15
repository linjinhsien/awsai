# AWS Awesome Day 2024 知識測驗

## Q1: Amazon S3 的特性 (選擇兩項)

- [ ] 全球文件系統 (A global file system)
- [x] 對象存儲 (An object store)
- [ ] 本地文件存儲 (A local file store)
- [ ] 網絡文件系統 (A network file system)
- [x] 持久性存儲系統 (A durable storage system)

> **解釋**：Amazon S3 是一個對象存儲服務，提供高持久性。它不是文件系統或網絡存儲。

---

## Q2: 允許客戶以折扣價購買未使用 EC2 容量的 AWS 服務

- [ ] 預留實例 (Reserved Instances)
- [ ] 按需實例 (On-Demand Instances)
- [ ] 專用實例 (Dedicated Instances)
- [x] 競價型實例 (Spot Instances)

> **解釋**：Spot 實例允許客戶利用 AWS 未使用的 EC2 容量，通常以較低的價格提供。

---

## Q3: 使用 AWS 雲對於在全球多個國家有客戶的公司的好處 (選擇兩項)

- [x] 公司可以在多個 AWS 區域部署應用程序以減少延遲。
- [ ] Amazon Translate 自動將第三方網站界面翻譯成多種語言。
- [x] Amazon CloudFront 在全球有多個邊緣位置以減少延遲。
- [ ] Amazon Comprehend 允許���戶構建可以用多種語言響應用戶請求的應用程序。

> **解釋**：這兩個選項都直接關於減少全球用戶訪問的延遲。

---

## Q4: 監控和接收有關 AWS 賬戶根用戶登錄事件的警報

- [x] Amazon CloudWatch
- [ ] AWS Config
- [ ] AWS Trusted Advisor
- [ ] AWS Identity and Access Management (IAM)

> **解釋**：CloudWatch 可以設置監控和警報，包括根用戶登錄事件。

---

## Q5: 識別允許無限制訪問用戶 AWS 資源的安全組的 AWS 服務

- [x] AWS Trusted Advisor
- [ ] AWS Config
- [ ] Amazon CloudWatch
- [ ] AWS CloudTrail

> **解釋**：Trusted Advisor 提供安全性檢查，包括識別過於寬松的安全組設置。

---

## 補充知識點

### Serverless（無服務器）
- 高可用性
- 高安全性
- 高效能

### Amazon Bedrock
- 可通過 API 進行串接
- 建議使用 SDK 進行程式開發

### 異地備援
- 網絡：使用 VPC 實現 Infrastructure as Code
- 計算：使用 EC2 和 CloudFormation
- 存儲：使用 S3 複製功能

### 遷移策略
- Rehost 和 Replatform 的差異

### AWS 安全架構
- 堡壘機：EC2 位於公有子網
- 應用服務器：位於私有子網

### 數據庫服務
- AWS Aurora：與 MySQL 和 PostgreSQL 兼容
- 跨 3 個可用區（AZ）部署
- DynamoDB：全託管 NoSQL 數據庫
- Redshift：數據倉庫解決方案

### 網絡
- NAT 網關：用於私有子網到公網的訪問

### AI 服務
- Amazon Q：屬於 Tools（SaaS 層）
- Amazon Bedrock with Claude：屬於 Platform 層
- Amazon Q 本質上也是調用 AWS Bedrock

### AWS SageMaker
- 自動化機器學習工作流程
- 支持多種機器學習算法和數據格式
- 與其他 AWS 服務集成
- 提供可視化工具和安全監控功能

以下是您提供的 `13q.md` 文件的美化排版版本。這個版本使用了標題、列表和引用格式來提高可讀性和結構性。



````markdown
# AWS AI Certification Exam Study Guide

## 1. 產品價格預測模型
**問題**: 一個組織正在開發一個模型來預測產品的價格，基於各種特徵，如大
小、重量、品牌和製造日期。哪種機器學習方法最適合這個任務？

**選項**:
- A. Classification
- B. Regression
- C. Clustering
- D. Dimensionality Reduction

**答案**: B. Regression  
**解釋**: 
- Clustering 將相似的數據點分組。
- Dimensionality Reduction 減少特徵數量。
- Regression 預測連續的數值。
- Classification 預測類別結果。

---

## 2. 人工智能核心原則
**問題**: 一家公司正在擴大其人工智能的使用。他們應優先考慮哪些核心原
則，以建立明確的指導方針、監督和責任？

**選項**:
- A. Bias Prevention
- B. Accuracy and Reliability
- C. Data Protection and Security
- D. Governance

**答案**: D. Governance  
**解釋**: Governance 涵蓋了管理 AI 系統的整體框架，包括政策、程序和決
策過程。其他選項也很重要，但對於監督來說，治理是最佳答案。

---

## 3. 責任 AI 實踐
**問題**: 一家公司開始在 AWS 上使用生成式 AI。為了確保負責任的 AI 實
踐，哪種工具可以為他們提供指導和信息？

**選項**:
- A. AWS Marketplace
- B. AWS AI Service Cards
- C. SageMaker
- D. Bedrock

**答案**: B. AWS AI Service Cards  
**解釋**: AI Service Cards 提供有關特定 AWS AI 服務的詳細信息，包括
其預期用例、限制和負責任的設計考慮。

---

## 4. 特徵工程的主要目的
**問題**: 特徵工程在機器學習中的主要目的是什麼？

**選項**:
- A. 確保模型的一致性能
- B. 評估模型的性能
- C. 收集和預處理數據特徵
- D. 轉換數據並為模型創建變量或特徵

**答案**: D. 轉換數據並為模型創建變量或特徵  
**解釋**: 這是特徵工程的主要目的。

---

## 5. 預測客戶流失的工具
**問題**: 一家小公司想使用機器學習來預測客戶流失，但他們缺乏專門的數據
科學團隊。哪種 AWS 工具可以幫助他們輕鬆構建模型，而無需大量編碼？

**選項**:
- A. Amazon SageMaker Jumpstart
- B. SageMaker Studio
- C. SageMaker Canvas
- D. SageMaker Data Wrangler

**答案**: C. SageMaker Canvas  
**解釋**: SageMaker Canvas 是一個可視化界面，用於構建機器學習模型，而
無需編寫代碼。

---

## 6. MLOps 的解釋
**問題**: 一家金融機構正在開發一個欺詐檢測模型。項目負責人宣布他們將使
用 MLOps。如何在此項目中解釋 MLOps？

**選項**:
- A. 用於可視化 ML 模型性能的工具
- B. 管理 ML 系統整個生命周期的一組實踐
- C. 部署和維護 ML 模型的過程
- D. 構建和訓練 ML 模型的框架

**答案**: B. 管理 ML 系統整個生命周期的一組實踐  
**解釋**: MLOps 涵蓋了整個生命周期，包括開發、部署、監控和維護。

---

## 7. 生成內容的 AWS 服務
**問題**: 一家公司希望使用現有的流行預訓練 AI 模型生成內容。他們的 AI 
專業知識有限，不想自己管理模型。哪種 AWS 服務最適合他們的需求？

**選項**:
- A. Amazon Textract
- B. Amazon Comprehend
- C. Amazon Bedrock
- D. Amazon SageMaker

**答案**: C. Amazon Bedrock  
**解釋**: Bedrock 是一個管理服務，提供對預訓練基礎模型的訪問，使其成為
希望利用 AI 而無需大量技術專業知識或基礎設施的公司的理想選擇。

---

## 8. API 調用日誌
**問題**: 一家公司需要記錄對 Amazon Bedrock 的 API 調用以符合合規
性，包括有關 API 調用、用戶和時間戳的詳細信息。哪種 AWS 服務可以協助此
操作？

**選項**:
- A. CloudTrail
- B. CloudWatch
- C. IAM
- D. Security Hub

**答案**: A. CloudTrail  
**解釋**: CloudTrail 可以記錄對 AWS 服務的 API 調用，包括 API 調用的
詳細信息、執行該操作的用戶和時間戳，適合用於跟踪合規性目的的 API 活
動。

---

## 9. 提高模型性能
**問題**: 一個數據科學團隊希望提高模型的性能。他們想增加用於訓練的數據
量和多樣性，並修改算法的學習率。哪種 ML 管道步驟的組合將滿足這些要求？

**選項**:
- A. Data Augmentation
- B. Model Monitoring
- C. Feature Engineering
- D. Hyperparameter Tuning

**答案**: A 和 D. Data Augmentation 和 Hyperparameter Tuning  
**解釋**: 
- Data Augmentation 是一種通過對現有數據點應用各種變換來人工增加數據
集大小和多樣性的技術。
- Hyperparameter Tuning 涉及調整學習率，這是一個關鍵參數，控制模型在
訓練過程中更新權重的速度。

---

## 10. 內容生成的道德準則
**問題**: 一家公司希望確保其 Amazon Bedrock 驅動的應用程序生成的內容
符合其道德準則，並避免有害或冒犯性的內容。哪種 AWS 服務可以幫助他們實
施這些保障措施？

**選項**:
- A. Amazon SageMaker
- B. Amazon Comprehend
- C. Textract
- D. Guardrails for Amazon Bedrock

**答案**: D. Guardrails for Amazon Bedrock  
**解釋**: Guardrails 確保基礎模型的負責任使用，符合組織的合規要求，通
過過濾和監控輸出來防止有害、偏見或不當內容的生成。

---

## 11. 敏感數據處理
**問題**: 您的公司正在對存儲在 S3 中的數據集進行機器學習模型訓練，該數
據集包含敏感的客戶信息。您如何確保在訓練模型之前刪除或匿名化數據中的任
何敏感信息？

**選項**:
- A. 使用 S3 加密保護靜態數據
- B. 使用 Amazon Macie 識別數據集中的敏感信息
- C. 使用 S3 訪問控制限制授權人員的訪問
- D. 實施數據掩碼技術以替換敏感信息

**答案**: B 和 D.  
**解釋**: 使用 Amazon Macie 識別數據集中的敏感信息，並實施數據掩碼技
術以替換敏感信息。其他選項有助於保護數據，但與刪除或匿名化敏感數據無
關。

---

## 12. 生成式 AI 的市場營銷
**問題**: 一家公司希望使用生成式 AI 為其產品創建市場營銷標語。為什麼該
公司應仔細審查所有生成的標語？

**選項**:
- A. 生成式 AI 可能生成過長且難以記住的標語
- B. 生成式 AI 可能難以捕捉公司的獨特品牌形象
- C. 生成式 AI 可能生成不當或誤導性的標語
- D. 生成式 AI 可能需要大量訓練數據來生成有效的標語

**答案**: C. 生成式 AI 可能生成不當或誤導性的標語  
**解釋**: 生成式 AI 有時可能生成偏見或冒犯性的內容，這可能是由於訓練數
據中存在的偏見。因此，人為監督對於確保生成標語的質量和適當性至關重要。

---

## 13. 機器學習模型的安全性
**問題**: 您的公司正在 EC2 實例上訓練機器學習模型。您擔心這些模型的安
全性，並希望識別基礎設施中的潛在漏洞。哪種 AWS 服務可以幫助您掃描 EC2 
實例的漏洞？

**選項**:
- A. AWS X-Ray
- B. Amazon CloudWatch
- C. Amazon Inspector
- D. AWS Config

**答案**: C. Amazon Inspector  
**解釋**: Amazon Inspector 是一項安全服務，幫助您識別 EC2 實例中的漏
洞。

---

## 14. 偏見類型
**問題**: 一個用於貸款批准的機器學習模型對城市地區的申請者表現更好，因
為訓練數據中包含更多來自城市地區的批准示例。這是什麼類型的偏見的例子？

**選項**:
- A. Sampling Bias
- B. Algorithm Bias
- C. Observer Bias
- D. Recency Bias

**答案**: A. Sampling Bias  
**解釋**: Sampling Bias 發生在訓練數據不代表所有情況時。

---

## 15. 生成式 AI 的應用
**問題**: 您正在構建一個文本摘要工具，哪種指標最適合衡量它捕捉原始文本
關鍵點的能力？

**選項**:
- A. BERT Score
- B. ROUGE
- C. Word Error Rate
- D. Bilingual Evaluation Understudy (BLEU)

**答案**: B. ROUGE  
**解釋**: ROUGE 專門設計用於評估文本摘要的質量。

---

## 16. 生成式任務的 AI 模型
**問題**: 您希望根據文本描述生成高細節的圖像，哪種專門設計用於生成任務
的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 17. 向量嵌入存儲
**問題**: 一家公司有一個系統從產品數據生成向量嵌入，他們希望提高查找相
似產品的速度和準確性。哪種 AWS 服務最適合此需求？

**選項**:
- A. Amazon S3
- B. Amazon DynamoDB
- C. Amazon OpenSearch Service
- D. Amazon Kendra

**答案**: C. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量嵌入的存儲和查詢。

---

這個版本的排版使得內容更加結構化和易於閱讀。每個問題都有清晰的標題、選
項和解釋，並且使用了分隔線來區分不同的問題。希望這樣的排版能夠幫助您更
好地理解和學習！

## 18. 自動化貸款處理
**問題**: 一家銀行每天收到大量貸款申請，貸款處理團隊手動提取這些申請中
的信息，這非常耗時。目標是使用 AI 工具自動化此過程。哪種 AWS 服務將對
此有幫助？

**選項**:
- A. Amazon Rekognition
- B. Amazon Textract
- C. Amazon Translate
- D. Amazon Transcribe

**答案**: B. Amazon Textract  
**解釋**: Amazon Textract 可以從掃描的文檔中提取文本，適合自動化提取
信息的需求。

---

## 19. 醫療預測模型
**問題**: 一家醫療公司希望開發一個機器學習模型，以預測患者發展糖尿病的
可能性，基於各種健康指標。以下哪個指標最適合評估模型的性能？

**選項**:
- A. Accuracy
- B. Precision
- C. F1 Score
- D. Recall
- E. Area Under ROC Curve (AUC)

**答案**: D 和 E. Recall 和 AUC  
**解釋**: 在這種情況下，Recall 和 AUC 是適合的指標，因為它們能夠有效
評估分類模型的性能。

---

## 20. 深度學習模型的轉移學習
**問題**: 一個組織在一個大型通用圖像數據集上訓練了一個深度學習模型，現
在希望將同一模型應用於分類醫療圖像，數據集較小。哪種機器學習技術最適合
這種情況？

**選項**:
- A. Reinforcement Learning
- B. Transfer Learning
- C. Supervised Learning
- D. Unsupervised Learning

**答案**: B. Transfer Learning  
**解釋**: Transfer Learning 是一種技術，重用預訓練模型以應對新的但相
關的任務，通常需要對新數據進行一些微調。

---

## 21. 安全的 VPC 連接
**問題**: 您正在 AWS 上構建一個機器學習模型，並希望安全地與第三方合作
夥伴共享它。哪種 AWS 服務可以幫助您在您的 VPC 和合作夥伴的 VPC 之間建
立私有連接，確保數據不暴露於公共互聯網？

**選項**:
- A. Direct Connect
- B. PrivateLink
- C. Transit Gateway
- D. VPN

**答案**: B. PrivateLink  
**解釋**: PrivateLink 使 VPC 之間或 AWS 服務之間的安全私有連接成為可
能，而不會將數據暴露於公共互聯網。

---

## 22. AWS 的共享責任模型
**問題**: 您正在使用 AWS SageMaker 在敏感客戶數據上訓練機器學習模型。
在 AWS 共享責任模型下，以下哪項主要是您的責任？

**選項**:
- A. 確保 AWS SageMaker 基礎設施的安全
- B. 保護 SageMaker 實例的底層操作系統
- C. 確保存儲在 S3 中的客戶數據的安全
- D. 修補 AWS SageMaker 軟件

**答案**: C. 確保存儲在 S3 中的客戶數據的安全  
**解釋**: 根據 AWS 共享責任模型，您作為 AWS 用戶主要負責客戶數據的安
全，包括為存儲在 S3 中的數據實施適當的安全措施和訪問控制。

---

## 23. 生成式 AI 的風險評估
**問題**: 在實施生成式 AI 安全範疇矩陣時，以下哪個因素應評估以確定與生
成式 AI 項目相關的風險水平？

**選項**:
- A. 模型的計算效率
- B. 用於訓練模型的數據的敏感性
- C. 推理延遲
- D. 模型中的參數數量

**答案**: B. 用於訓練模型的數據的敏感性  
**解釋**: 敏感數據的使用是安全的關鍵因素，其他選項雖然重要，但不會直接
影響安全性。

---

## 24. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任
務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 18. 自動化貸款處理
**問題**: 一家銀行每天收到大量貸款申請，貸款處理團隊手動提取這些申請中的信息，這非常耗時。目標是使用 AI 工具自動化此過程。哪種 AWS 服務將對此有幫助？

**選項**:
- A. Amazon Rekognition
- B. Amazon Textract
- C. Amazon Translate
- D. Amazon Transcribe

**答案**: B. Amazon Textract  
**解釋**: Amazon Textract 可以從掃描的文檔中提取文本，適合自動化提取信息的需求。

---

## 19. 醫療預測模型
**問題**: 一家醫療公司希望開發一個機器學習模型，以預測患者發展糖尿病的可能性，基於各種健康指標。以下哪個指標最適合評估模型的性能？

**選項**:
- A. Accuracy
- B. Precision
- C. F1 Score
- D. Recall
- E. Area Under ROC Curve (AUC)

**答案**: D 和 E. Recall 和 AUC  
**解釋**: 在這種情況下，Recall 和 AUC 是適合的指標，因為它們能夠有效評估分類模型的性能。

---

## 20. 深度學習模型的轉移學習
**問題**: 一個組織在一個大型通用圖像數據集上訓練了一個深度學習模型，現在希望將同一模型應用於分類醫療圖像，數據集較小。哪種機器學習技術最適合這種情況？

**選項**:
- A. Reinforcement Learning
- B. Transfer Learning
- C. Supervised Learning
- D. Unsupervised Learning

**答案**: B. Transfer Learning  
**解釋**: Transfer Learning 是一種技術，重用預訓練模型以應對新的但相關的任務，通常需要對新數據進行一些微調。

---

## 21. 安全的 VPC 連接
**問題**: 您正在 AWS 上構建一個機器學習模型，並希望安全地與第三方合作夥伴共享它。哪種 AWS 服務可以幫助您在您的 VPC 和合作夥伴的 VPC 之間建立私有連接，確保數據不暴露於公共互聯網？

**選項**:
- A. Direct Connect
- B. PrivateLink
- C. Transit Gateway
- D. VPN

**答案**: B. PrivateLink  
**解釋**: PrivateLink 使 VPC 之間或 AWS 服務之間的安全私有連接成為可能，而不會將數據暴露於公共互聯網。

---

## 22. AWS 的共享責任模型
**問題**: 您正在使用 AWS SageMaker 在敏感客戶數據上訓練機器學習模型。在 AWS 共享責任模型下，以下哪項主要是您的責任？

**選項**:
- A. 確保 AWS SageMaker 基礎設施的安全
- B. 保護 SageMaker 實例的底層操作系統
- C. 確保存儲在 S3 中的客戶數據的安全
- D. 修補 AWS SageMaker 軟件

**答案**: C. 確保存儲在 S3 中的客戶數據的安全  
**解釋**: 根據 AWS 共享責任模型，您作為 AWS 用戶主要負責客戶數據的安全，包括為存儲在 S3 中的數據實施適當的安全措施和訪問控制。

---

## 23. 生成式 AI 的風險評估
**問題**: 在實施生成式 AI 安全範疇矩陣時，以下哪個因素應評估以確定與生成式 AI 項目相關的風險水平？

**選項**:
- A. 模型的計算效率
- B. 用於訓練模型的數據的敏感性
- C. 推理延遲
- D. 模型中的參數數量

**答案**: B. 用於訓練模型的數據的敏感性  
**解釋**: 敏感數據的使用是安全的關鍵因素，其他選項雖然重要，但不會直接影響安全性。

---

## 24. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

這個版本的排版使得內容更加結構化和易於閱讀。每個問題都有清晰的標題、選項和解釋，並且使用了分隔線來區分不同的問題。希望這樣的排版能夠幫助您更好地理解和學習！如果還有其他部分需要編排或美化，請隨時告訴我！

## 25. 向量搜索的 AWS 服務
**問題**: 一家公司有一個系統從產品數據生成向量嵌入，他們希望提高查找相似產品的速度和準確性。哪種 AWS 服務最適合此需求？

**選項**:
- A. Amazon OpenSearch Service
- B. Amazon S3
- C. Amazon DynamoDB
- D. Amazon Kendra

**答案**: A. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量嵌入的存儲和查詢，適合用於相似性搜索。

---

## 26. 自然語言處理的基本單位
**問題**: 在自然語言處理的上下文中，以下哪一項是用來表示單詞或子詞的基本單位？

**選項**:
- A. Token
- B. Vector Embedding
- C. N-gram
- D. Vocabulary

**答案**: A. Token  
**解釋**: Token 是文本的基本單位，用於表示單詞或子詞，經過標記化處理後生成。

---

## 27. 機器學習模型的透明度
**問題**: 一名開發人員正在創建一個 AI 系統來預測客戶流失。為了確保透明度，他們需要記錄有關模型的關鍵細節。哪種 AWS 工具最適合此任務？

**選項**:
- A. Amazon SageMaker Clarify
- B. AWS AI Service Cards
- C. SageMaker Model Cards
- D. SageMaker Jumpstart

**答案**: C. SageMaker Model Cards  
**解釋**: SageMaker Model Cards 用於記錄和分享有關機器學習模型的詳細信息，確保透明度。

---

## 28. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 29. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 30. 機器學習模型的性能評估
**問題**: 一個機器學習模型在訓練數據上表現良好，但在新數據上表現不佳。可能的問題是什麼？

**選項**:
- A. Overfitting
- B. Underfitting
- C. Insufficient Training Data
- D. Poor Data Quality

**答案**: A. Overfitting  
**解釋**: Overfitting 是指模型過於複雜，學習了訓練數據的細節，導致無法對新數據進行良好的泛化。

---

## 31. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 32. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 33. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 34. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

這個版本的排版使得內容更加結構化和易於閱讀。每個問題都有清晰的標題、選項和解釋，並且使用了分隔線來區分不同的問題。希望這樣的排版能夠幫助您更好地理解和學習！如果還有其他部分需要編排或美化，請隨時告訴我！

## 35. 機器學習模型的調整
**問題**: 一個數據科學團隊希望提高模型的性能。他們想增加用於訓練的數據量和多樣性，並調整算法的學習率。哪種 ML 管道步驟的組合將滿足這些要求？

**選項**:
- A. Data Augmentation
- B. Model Monitoring
- C. Feature Engineering
- D. Hyperparameter Tuning

**答案**: A 和 D. Data Augmentation 和 Hyperparameter Tuning  
**解釋**: 
- Data Augmentation 是一種通過對現有數據點應用各種變換來人工增加數據集大小和多樣性的技術。
- Hyperparameter Tuning 涉及調整學習率，這是一個關鍵參數，控制模型在訓練過程中更新權重的速度。

---

## 36. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 37. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 38. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 39. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 40. 機器學習模型的性能評估
**問題**: 一個機器學習模型在訓練數據上表現良好，但在新數據上表現不佳。可能的問題是什麼？

**選項**:
- A. Overfitting
- B. Underfitting
- C. Insufficient Training Data
- D. Poor Data Quality

**答案**: A. Overfitting  
**解釋**: Overfitting 是指模型過於複雜，學習了訓練數據的細節，導致無法對新數據進行良好的泛化。

---

## 41. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 42. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 43. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 44. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

這個版本的排版使得內容更加結構化和易於閱讀。每個問題都有清晰的標題、選項和解釋，並且使用了分隔線來區分不同的問題。希望這樣的排版能夠幫助您更好地理解和學習！如果還有其他部分需要編排或美化，請隨時告訴我！

## 45. 機器學習模型的調整
**問題**: 一個數據科學團隊希望提高模型的性能。他們想增加用於訓練的數據量和多樣性，並調整算法的學習率。哪種 ML 管道步驟的組合將滿足這些要求？

**選項**:
- A. Data Augmentation
- B. Model Monitoring
- C. Feature Engineering
- D. Hyperparameter Tuning

**答案**: A 和 D. Data Augmentation 和 Hyperparameter Tuning  
**解釋**: 
- Data Augmentation 是一種通過對現有數據點應用各種變換來人工增加數據集大小和多樣性的技術。
- Hyperparameter Tuning 涉及調整學習率，這是一個關鍵參數，控制模型在訓練過程中更新權重的速度。

---

## 46. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 47. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 48. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 49. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 50. 機器學習模型的性能評估
**問題**: 一個機器學習模型在訓練數據上表現良好，但在新數據上表現不佳。可能的問題是什麼？

**選項**:
- A. Overfitting
- B. Underfitting
- C. Insufficient Training Data
- D. Poor Data Quality

**答案**: A. Overfitting  
**解釋**: Overfitting 是指模型過於複雜，學習了訓練數據的細節，導致無法對新數據進行良好的泛化。

---

## 51. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 52. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 53. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 54. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 55. 機器學習模型的調整
**問題**: 一個數據科學團隊希望提高模型的性能。他們想增加用於訓練的數據量和多樣性，並調整算法的學習率。哪種 ML 管道步驟的組合將滿足這些要求？

**選項**:
- A. Data Augmentation
- B. Model Monitoring
- C. Feature Engineering
- D. Hyperparameter Tuning

**答案**: A 和 D. Data Augmentation 和 Hyperparameter Tuning  
**解釋**: 
- Data Augmentation 是一種通過對現有數據點應用各種變換來人工增加數據集大小和多樣性的技術。
- Hyperparameter Tuning 涉及調整學習率，這是一個關鍵參數，控制模型在訓練過程中更新權重的速度。

---

## 56. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 57. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 58. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 59. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 60. 機器學習模型的性能評估
**問題**: 一個機器學習模型在訓練數據上表現良好，但在新數據上表現不佳。可能的問題是什麼？

**選項**:
- A. Overfitting
- B. Underfitting
- C. Insufficient Training Data
- D. Poor Data Quality

**答案**: A. Overfitting  
**解釋**: Overfitting 是指模型過於複雜，學習了訓練數據的細節，導致無法對新數據進行良好的泛化。

---

## 61. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 62. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 63. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 64. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---


## 65. 機器學習模型的調整
**問題**: 一個數據科學團隊希望提高模型的性能。他們想增加用於訓練的數據量和多樣性，並調整算法的學習率。哪種 ML 管道步驟的組合將滿足這些要求？

**選項**:
- A. Data Augmentation
- B. Model Monitoring
- C. Feature Engineering
- D. Hyperparameter Tuning

**答案**: A 和 D. Data Augmentation 和 Hyperparameter Tuning  
**解釋**: 
- Data Augmentation 是一種通過對現有數據點應用各種變換來人工增加數據集大小和多樣性的技術。
- Hyperparameter Tuning 涉及調整學習率，這是一個關鍵參數，控制模型在訓練過程中更新權重的速度。

---

## 66. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 67. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 68. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 69. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---

## 70. 機器學習模型的性能評估
**問題**: 一個機器學習模型在訓練數據上表現良好，但在新數據上表現不佳。可能的問題是什麼？

**選項**:
- A. Overfitting
- B. Underfitting
- C. Insufficient Training Data
- D. Poor Data Quality

**答案**: A. Overfitting  
**解釋**: Overfitting 是指模型過於複雜，學習了訓練數據的細節，導致無法對新數據進行良好的泛化。

---

## 71. 生成式 AI 的應用
**問題**: 您希望生成高質量的圖像，基於文本描述，哪種專門設計用於生成任務的 AI 模型最適合此任務？

**選項**:
- A. Generative Adversarial Networks (GANs)
- B. Recurrent Neural Networks (RNNs)
- C. Convolutional Neural Networks (CNNs)
- D. Stable Diffusion

**答案**: D. Stable Diffusion  
**解釋**: Stable Diffusion 是專門設計用於生成任務的模型，如圖像生成。

---

## 72. 向量嵌入的存儲
**問題**: 一家公司需要選擇一種服務來存儲和查詢向量嵌入。哪種 AWS 服務最適合此需求？

**選項**:
- A. Glue Data Catalog
- B. Amazon S3
- C. Redshift
- D. OpenSearch Service

**答案**: D. Amazon OpenSearch Service  
**解釋**: OpenSearch Service 支持向量數據的存儲和查詢，適合用於向量嵌入的應用。

---

## 73. 數據隱私的考量
**問題**: 您的公司正在使用 AWS SageMaker 訓練機器學習模型，並希望確保數據隱私。以下哪種措施最能保護敏感數據？

**選項**:
- A. 使用 S3 加密保護數據
- B. 使用 Amazon Macie 識別敏感數據
- C. 實施數據掩碼技術
- D. 限制數據訪問權限

**答案**: C. 實施數據掩碼技術  
**解釋**: 數據掩碼技術可以有效地替換或隱藏敏感信息，保護數據隱私。

---

## 74. 生成式 AI 的風險管理
**問題**: 在實施生成式 AI 項目時，以下哪個因素應被評估以確定風險水平？

**選項**:
- A. 模型的計算效率
- B. 數據的敏感性
- C. 推理延遲
- D. 模型的參數數量

**答案**: B. 數據的敏感性  
**解釋**: 數據的敏感性是評估生成式 AI 項目風險的關鍵因素。

---
