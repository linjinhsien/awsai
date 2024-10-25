# AWS Awesome Day 2024 知識測驗

## Q1: Amazon S3 的特性 (選擇兩項)

- [ ] A global file system
- [x] An object store
- [ ] A local file store
- [ ] A network file system
- [x] A durable storage system

**解釋**: Amazon S3 是一個對象存儲服務,提供高持久性。它不是文件系統或網絡存儲。

## Q2: 允許客戶以折扣價購買未使用 EC2 容量的 AWS 服務

- [ ] Reserved Instances
- [ ] On-Demand Instances
- [ ] Dedicated Instances
- [x] Spot Instances

**解釋**: Spot 實例允許客戶利用 AWS 未使用的 EC2 容量,通常以較低的價格提供。

## Q3: 使用 AWS 雲對於在全球多個國家有客戶的公司的好處 (選擇兩項)

- [x] 公司可以在多個 AWS 區域部署應用程序以減少延遲。
- [ ] Amazon Translate 自動將第三方網站界面翻譯成多種語言。
- [x] Amazon CloudFront 在全球有多個邊緣位置以減少延遲。
- [ ] Amazon Comprehend 允許用戶構建可以用多種語言響應用戶請求的應用程序。

**解釋**: 這兩個選項都直接關於減少全球用戶訪問的延遲。

## Q4: 監控和接收有關 AWS 賬戶根用戶登錄事件的警報

- [x] Amazon CloudWatch
- [ ] AWS Config
- [ ] AWS Trusted Advisor
- [ ] AWS Identity and Access Management (IAM)

**解釋**: CloudWatch 可以設置監控和警報,包括根用戶登錄事件。

## Q5: 識別允許無限制訪問用戶 AWS 資源的安全組的 AWS 服務

- [x] AWS Trusted Advisor
- [ ] AWS Config
- [ ] Amazon CloudWatch
- [ ] AWS CloudTrail

**解釋**: Trusted Advisor 提供安全性檢查,包括識別過於寬松的安全組設置。

知識測驗 Q1 (共五題): Which of the functionalities are characteristics of Amazon S3? (Select TWO.)
A global file system
 An object store
A local file store
A network file system
A durable storage system
知識測驗 Q2 (共五題): Which AWS service allows customers to purchase unused Amazon EC2 capacity at an often discounted rate?
Reserved Instances
On-Demand Instances
Dedicated Instances
Spot Instances
知識測驗 Q3 (共五題): What are benefits of using the AWS Cloud for companies with customers in many countries around the world? (Select TWO.)
Companies can deploy applications in multiple AWS Regions to reduce latency.
Amazon Translate automatically translates third-party website interfaces into multiple languages.
Amazon CloudFront has multiple edge locations around the world to reduce latency.
Amazon Comprehend allow users to build applications that can respond to user requests in many languages.
知識測驗 Q4 (共五題): A company needs to monitor and receive alerts about AWS Management Console sign-in events that involve the AWS account root user. Which AWS service can the company use to meet these requirements?
Amazon CloudWatch
AWS Config
AWS Trusted Advisor
AWS Identity and Access Management (IAM)
知識測驗 Q5 (共五題): Which AWS service identifies security groups that allow unrestricted access to a user's AWS resources?
AWS Trusted Advisor
AWS Config
Amazon CloudWatch
AWS CloudTrail
我很高興為您解答這些AWS知識測驗題目。讓我們逐一分析每個問題:

1. Amazon S3的特性 (選擇兩項):
   正確答案: 
   - 對象存儲 (An object store)
   - 持久性存儲系統 (A durable storage system)
   
   解釋: Amazon S3是一個對象存儲服務,提供高持久性。它不是文件系統或網絡存儲。

2. 允許客戶以折扣價購買未使用EC2容量的AWS服務:
   正確答案: Spot Instances (競價型實例)
   
   解釋: Spot實例允許客戶利用AWS未使用的EC2容量,通常以較低的價格提供。

3. 使用AWS雲對於在全球多個國家有客戶的公司的好處 (選擇兩項):
   正確答案:
   - 公司可以在多個AWS區域部署應用程序以減少延遲。
   - Amazon CloudFront在全球有多個邊緣位置以減少延遲。
   
   解釋: 這兩個選項都直接關於減少全球用戶訪問的延遲。

4. 監控和接收有關AWS賬戶根用戶登錄事件的警報:
   正確答案: Amazon CloudWatch
   
   解釋: CloudWatch可以設置監控和警報,包括根用戶登錄事件。

5. 識別允許無限制訪問用戶AWS資源的安全組的AWS服務:
   正確答案: AWS Trusted Advisor
   
   解釋: Trusted Advisor提供安全性檢查,包括識別過於寬松的安全組設置。

這些問題涵蓋了AWS的多個關鍵服務和概念,包括存儲、計算、全球基礎設施、監控和安全性。希望這些解答對您有所幫助!




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

### 資料庫服務
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
seveless 
高可用
高安全
高效能


bedrock  串接api

程式開發 建議用sdk


異地備援   

網路 vpc  infrastructure as a code

EC2  cloudformation

算力

s3 replication 

rehost  replatform 的差異

aws 堡壘機   ec2 公子網  私有子網

aws aurora  mysql  postgres 相容
3個az 


dynamodb  no sql  全託管

redshift  資料倉儲

nat 私有子網到公開子網


Amazon Q 屬Tools (SaaS層)，Aamazon Bedrok with Claude (屬Platform層)，簡單理解Amazon Q 本質上也是調用AWS Bedrock



AWS SageMaker 是 Amazon Web Services（AWS）的一個機器學習（Machine Learning）服務平台，旨在幫助開發者快速創建、訓練和部署機器學習模型。下面是 SageMaker 的主要功能：

自動化機器學習工作流程：SageMaker 可以自動化機器學習的工作流程，包括資料準備、模型訓練、模型評估和模型部署等步驟。

自動化資料準備：SageMaker 可以自動將資料轉換為適合機器學習的格式，並且可以自動處理資料缺失、資料清潔和資料轉換等問題。

自動化模型訓練：SageMaker 可以自動化機器學習模型的訓練過程，包括選擇適合的模型算法、設定模型參數和調整模型 hyperparameters 等。

自動化模型評估：SageMaker 可以自動化機器學習模型的評估過程，包括評估模型的準確性、召回率和 F1 分數等指標。

自動化模型部署：SageMaker 可以自動化機器學習模型的部署過程，包括將模型部署到生產環境、設定模型的實時預測和監控等。

支持多種機器學習算法：SageMaker 支持多種機器學習算法，包括線性回歸、 decision tree、 random forest、支持向量機、神經網路等。

支持多種資料格式：SageMaker 支持多種資料格式，包括 CSV、 JSON、 Parquet 等。

集成 AWS 服務：SageMaker 可以與其他 AWS 服務集成，例如 Amazon S3、 Amazon DynamoDB、 Amazon Redshift 等。

提供可視化工具：SageMaker 提供了可視化工具，例如 SageMaker Notebook Instance、 SageMaker Studio 等，可以幫助開發者更好地理解和 debug 機器學習模型。

提供安全和監控功能：SageMaker 提供了安全和監控功能，例如身份驗證、授權、監控模型性能和預測結果等，可以幫助開發者保護機器學習模型和資料。

總的來說，AWS SageMaker 是一個功能強大且易於使用的機器學習服務平台，可以幫助開發者快速創建、訓練和部署機器學習模型。


I'd be happy to help you explain the technical terms that appear in the text. Here's a breakdown of the key terms:

1. **機器學習 (Machine Learning)**: Machine learning is a subfield of artificial intelligence that involves training algorithms to learn from data and make predictions or decisions.

2. **自動化 (Automation)**: Automation refers to the process of automating tasks or workflows, in this case, the machine learning workflow.

3. **資料準備 (Data Preparation)**: Data preparation involves cleaning, transforming, and preparing data for use in machine learning models.

4. **模型訓練 (Model Training)**: Model training involves training a machine learning model on a dataset to learn patterns and relationships.

5. **模型評估 (Model Evaluation)**: Model evaluation involves assessing the performance of a machine learning model on a test dataset.

6. **模型部署 (Model Deployment)**: Model deployment involves deploying a trained machine learning model to a production environment.

7. **機器學習算法 (Machine Learning Algorithm)**: Machine learning algorithms are the mathematical formulas used to train and make predictions with machine learning models.

8. **資料格式 (Data Format)**: Data format refers to the structure and organization of data, such as CSV, JSON, or Parquet.

9. **AWS 服務 (AWS Service)**: AWS services refer to the various cloud computing services offered by Amazon Web Services, such as Amazon S3, Amazon DynamoDB, and Amazon Redshift.

10. **可視化工具 (Visualization Tool)**: Visualization tools are software applications that help users visualize and understand complex data and machine learning models.

11. **安全和監控功能 (Security and Monitoring Function)**: Security and monitoring functions refer to the measures taken to protect machine learning models and data from unauthorized access, tampering, or other security threats.

12. **身份驗證 (Authentication)**: Authentication refers to the process of verifying the identity of a user or system.

13. **授權 (Authorization)**: Authorization refers to the process of controlling access to resources or systems based on a user's identity or role.

14. **監控 (Monitoring)**: Monitoring refers to the process of tracking and analyzing the performance and behavior of machine learning models and systems.

I hope this helps! Let me know if you have any further questions.

