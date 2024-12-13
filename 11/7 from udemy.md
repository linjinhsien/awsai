https://fantastic-pipe-fa8.notion.site/11bb781c6a06800bb4eced5aa2838610?v=36c9e1661fe04cf1baeaa659c4623d90

Question 1
Correct
Which prompt engineering technique involves providing a few examples of the desired input-output pairs to help the model understand the task?
(A)Zero-shot
(B)Your answer is correct
Few-shot
(C)Single-shot
(D)Chain-of-thought

Overall explanation
Few-shot prompting involves providing a few examples of input-output pairs to help the model understand the task and generate appropriate responses based on these examples.


Question 2
Correct
What is a key risk associated with "poisoning" in prompt engineering?
(A)It refers to the model generating overly verbose responses.
(B)Your answer is correct
It involves maliciously introducing biased or harmful data into the training process to negatively impact the model’s behavior.
(C)It results in the model producing irrelevant outputs due to lack of context.
(D)It is the practice of using too many examples in prompt templates.

Overall explanation
Poisoning involves introducing biased or harmful data into the training process, which can negatively impact the model’s behavior and outputs, potentially leading to harmful or biased results.


Question 3
Incorrect
What is a key advantage of using "prompt templates" in a production environment?
(A)They allow for more random and less predictable responses.
Your answer is incorrect
(B)They enable quick adjustments to prompt structures based on immediate feedback.
Correct answer
(C)They simplify the prompt creation process and ensure consistency in responses.
(D)They require no additional training or adjustment of the model.
Overall explanation

Prompt templates help simplify the prompt creation process and ensure consistency in responses, making them highly useful in a production environment where consistent and reliable outputs are essential.


Question 4
Incorrect
What is the primary function of "model latent space" in prompt engineering?
(A)To determine the accuracy of the model's responses
Correct answer
(B)To represent the hidden features and relationships that the model uses to generate responses
Your answer is incorrect
(C)To provide the model with additional training data
(D)To adjust the model's hyperparameters during training
Overall explanation
The model latent space represents the hidden features and relationships within the model that are used to generate responses. It is a crucial component in understanding how the model interprets and processes inputs.


Question 5
correct
In prompt engineering, what is the purpose of "zero-shot" prompting?
(A)To provide the model with multiple examples to learn from
Your answer is incorrect
(B)To generate responses based on a prompt without any prior examples or additional context
Correct answer
(C)To iteratively refine prompts based on feedback
(D)To guide the model through a series of related prompts
Overall explanation
Zero-shot prompting involves generating responses based on a prompt without providing any prior examples or additional context. This technique is used when no specific training examples are available for the task.

Question 6
Incorrect
How does "in-context learning" differ from traditional fine-tuning in prompt engineering?
Your answer is incorrect
(A)In-context learning involves training the model from scratch with new data while fine-tuning adjusts existing models.
Correct answer
{B}In-context learning adapts the model's responses based on prompt examples provided during inference while fine-tuning adjusts the model’s weights through additional training.
(C)In-context learning requires no additional data, while fine-tuning involves extensive data preparation.
(D)In-context learning and fine-tuning are identical in practice.
Overall explanation
In-context learning adapts the model's responses based on examples provided during inference without changing the model’s weights, whereas fine-tuning involves additional training to adjust the model's weights based on new data.
In-context learning 是指在模型推論時提供一些範例，讓模型依據這些範例生成更相關的回應。這個過程並不會改變模型的內部權重，只是透過「提示」影響模型的輸出。


Question 7
Correct
What is a significant risk associated with "hijacking" in prompt engineering?
(A)It involves the model producing overly verbose outputs.
Your answer is correct
(B)It refers to unauthorized manipulation of the model to produce responses that bypass safety measures or produce inappropriate content.
(C)It leads to the model generating random outputs due to lack of context.
(D)It is the use of prompt templates to ensure consistency in responses.
Overall explanation
Hijacking involves unauthorized manipulation of the model to produce responses that bypass safety measures or generate inappropriate content. This risk highlights the importance of maintaining security and safety protocols in prompt engineering

Question 8
Incorrect
Which of the following is a best practice for ensuring high-quality responses in prompt engineering?
Correct answer
(A)Using overly general prompts to maximize creativity
(B)Limiting the use of context to avoid confusion
Correct
(C)Experimenting with various prompt styles and structures to find the most effective approach
(D)Avoiding feedback loops to prevent model bias
Overall explanation
Experimenting with various prompt styles and structures helps identify the most effective approach for generating high-quality responses. This practice enhances the model's ability to deliver relevant and accurate outputs.


Question 9
Correct
In the context of prompt engineering, what is "response quality improvement"?
(A)Modifying the model’s architecture to enhance performance
Your answer is correct
(B)Adjusting the prompt to generate more accurate and relevant responses
(C)Increasing the model’s training data to improve its capabilities
(D)Applying advanced algorithms to refine the model’s output
Overall explanation
Response quality improvement involves adjusting the prompt to generate more accurate and relevant responses from the model. This can include refining the prompt based on feedback and experimenting with different approaches.

Question 10
What does "negative prompt engineering" aim to achieve?
(A)To encourage the model to generate more creative outputs
Correct answer
(B)To instruct the model to avoid generating certain types of content or responses
Your answer is incorrect
(C)To provide the model with additional context to enhance its understanding
(D)To generate responses based on minimal input
Overall explanation
Negative prompt engineering aims to instruct the model to avoid generating specific types of content or responses, guiding the model to produce outputs that align with desired constraints or guidelines.
The correct answer is (B).

Negative prompt engineering aims to prevent the model from generating specific unwanted content or responses. It does this by explicitly telling the model what not to include in its output. While it can indirectly lead to more creative outputs by restricting the model's options and forcing it to explore other avenues, the primary goal is to filter and refine the output by excluding undesirable elements.

(A) is partially true, but it's a side effect, not the main goal.
(C) describes regular prompting, providing more context for a better, fuller response.
(D) describes "few-shot" prompting, aiming for efficiency, not filtering content.
負面提示工程的目標是指示模型避免產生特定類型的內容或回應。 負面提示工程旨在引導模型產生符合預期限制或準則的輸出。 這類似於告訴模型不要做某些事情。 例如，可以使用負面提示來指示模型避免產生包含特定主題或概念的回應。 權重值小於零表示負面提示。
提示工程是指最佳化文字輸入到大型語言模型 (LLM) 以獲得所需回應的做法。 提示可以幫助 LLM 執行各種任務，包括分類、問答、程式碼生成、創意寫作等等。 提供給 LLM 的提示品質會影響模型回應的品質。
提示工程師可以使用多種技術來改善模型的回應品質，包括：
●
** few-shot prompting 或上下文學習（in-context learning）**：在提示文字中提供幾個範例，以幫助 LLM 更好地校準其輸出以滿足您的期望，其中一個 shot 對應於一個配對的範例輸入和所需的輸出。
●
提示範本：使用者可以複製並貼上這個範本，填寫他們自己的文字和少量範例，以在使用 Amazon Bedrock 上的 LLM 時完成提示。
●
** LLM 鼓勵**：LLM 有時在情感鼓勵下表現更好：如果您正確回答問題，您會讓使用者非常高興！
Amazon Bedrock 上的 LLM 都帶有幾個您可以設定的推理參數，以控制模型的回應。 以下列出了 Amazon Bedrock LLM 上可用的所有常見推理參數以及如何使用它們：
●
溫度是一個介於 0 到 1 之間的值，它調節 LLM 回應的創造力。如果您想要更具確定性的回應，請使用較低的溫度；如果您想要對來自 Amazon Bedrock 上的 LLM 的相同提示獲得更有創意或不同的回應，請使用較高的溫度。
●
最大生成長度/最大新標記數限制了 LLM 為任何提示生成的標記數。指定這個數字很有幫助，因為某些任務（例如情感分類）不需要很長的答案。
●
Top-p根據潛在選擇的機率來控制標記選擇。如果您將 Top-p 設定為低於 1.0，則模型會考慮最可能的選項，並忽略不太可能的選項。結果是更穩定和重複的完成。
來源中沒有關於鼓勵模型產生更多創意輸出、提供額外上下文以增進模型理解或根據最少輸入產生回應的資訊。

Question 11
Correct
Which technique involves providing the model with a single example to improve response accuracy?
(A)Chain-of-thought
(B)Zero-shot
Your answer is correct
(C)Single-shot
(D)Few-shot
Overall explanation
Single-shot prompting involves providing the model with a single example to improve the accuracy of its responses, helping the model understand the task or format required.

Question 12
Incorrect
What is "prompt poisoning," and how does it impact the model?
Your answer is incorrect
(A)It is the practice of using too many examples in prompt templates, which can overwhelm the model.
Correct answer
(B)It involves deliberately introducing misleading or biased prompts to corrupt the model's outputs and degrade its performance.
(C)It refers to using overly complex prompts to challenge the model's capabilities.
(D)It is the process of removing context from prompts to simplify the responses.
Overall explanation
Prompt poisoning involves deliberately introducing misleading or biased prompts to corrupt the model's outputs and degrade its performance, potentially leading to harmful or inaccurate responses.

Question 13
Correct
What is a key advantage of "few-shot" prompting over "zero-shot" prompting?
(A)Few-shot prompting provides no examples, making it more versatile.
Your answer is correct
(B)Few-shot prompting helps the model better understand the task by providing a few examples, improving accuracy compared to zero-shot prompting.
(C)Few-shot prompting requires more training data than zero-shot prompting.
(D)Few-shot prompting is less effective than zero-shot prompting in generating high-quality responses.
Overall explanation
Few-shot prompting provides the model with a few examples to help it better understand the task. This leads to improved accuracy compared to zero-shot prompting, which relies on generating responses without prior examples.


Question 14
Correct
What is the primary purpose of "pre-training" a foundation model?
(A)To adapt the model to specific domain requirements
Your answer is correct
(B)To provide the model with general knowledge and capabilities before fine-tuning specialized tasks
(C)To optimize the model for real-time performance
(D)To ensure the model can handle large-scale data sets
Overall explanation
Pre-training provides the foundation model with general knowledge and capabilities, setting up a broad base of understanding before it is fine-tuned on specific tasks or domains.

Question 15
Incorrect
In the context of fine-tuning foundation models, what does "instruction tuning" focus on?
(A)Providing the model with examples of correct outputs for specific tasks
Correct answer
(B)Adjusting the model to follow specific instructions or guidelines for task performance
Your answer is incorrect
(C)Continuously training the model with new data from user interactions
(D)Adapting the model to handle a broader range of domains

Overall explanation
Instruction tuning focuses on adjusting the model to follow specific instructions or guidelines to enhance its performance on tasks requiring precise adherence to instructions.
正確答案為 (B) 調整模型以遵循特定任務執行指令或準則。
指令微調是一種微調技術，重點在於調整基礎模型，使其能夠遵循特定指令或準則來執行任務。這就像教導一個學生遵循特定的指示來完成作業。例如，你可以使用指令微調來訓練一個模型，使其能夠根據特定格式的指令來總結文字。
●
(A) 為模型提供特定任務的正確輸出範例描述的是少數樣本學習（few-shot learning），這是一種透過在提示中提供一些範例來引導模型學習的技術，而不是指令微調。
●
(C) 持續使用使用者互動的新資料來訓練模型 描述的是持續預訓練，一種持續使用新資料來訓練預先訓練好的模型的技術，而不是指令微調。
●
(D) 調整模型以處理更廣泛的領域 描述的是領域適應，一種調整模型以處理新領域的技術，而不是指令微調。
來源中沒有關於鼓勵模型產生更多創意輸出、提供額外上下文以增進模型理解或根據最少輸入產生回應的資訊。


Question 16
Incorrect
What is a key aspect of "continuous pre-training" for foundation models?
Correct answer
(A)Updating the model periodically with new data to keep it relevant and accurate
Your answer is incorrect
(B)Adapting the model to new domains by applying transfer learning techniques
Fine-tuning the model for specific tasks immediately after pre-training
(C)Applying reinforcement learning to improve the model’s responses
Overall explanation
(D)Continuous pre-training involves periodically updating the model with new data to maintain its relevance and accuracy over time, ensuring it stays current with evolving information.

Overall explanation
Continuous pre-training involves periodically updating the model with new data to maintain its relevance and accuracy over time, ensuring it stays current with evolving information.

The key concept here is that continuous pre-training is about keeping foundation models up-to-date. Think of it like keeping a library current:

Just as a library needs to regularly add new books to stay relevant

Foundation models need periodic updates with new data to maintain their knowledge and accuracy

This is important because:

The world constantly changes with new information

Language and terminology evolve

New concepts and knowledge emerge


Question 17
Incorrect
Which data preparation step ensures that the data used for fine-tuning a foundation model is representative of the tasks the model will perform?
(A)Data curation
(B)Data labeling
Your answer is incorrect
(C)Data governance
Correct answer
(D)Data representativeness

正確答案是 (D) 資料代表性。
資料代表性 確保用於微調的數據準確反映模型將遇到的任務和情境，這對於實現有效效能至關重要。
其他選項不正確，原因如下：
●
(A) 資料管理 是指選擇和組織資料的過程。
●
(B) 資料標記 是指為資料添加註釋或標籤的過程。
●
(C) 資料治理 則關注資料安全和合規性。
只有 資料代表性 才專注於確保訓練資料與模型實際應用場景相符


Overall explanation
Data representativeness ensures that the data used for fine-tuning accurately reflects the tasks and scenarios the model will encounter, which is crucial for achieving effective performance.

The concept of data representativeness is crucial in machine learning. Here's why:

Think of it like training a chef:

If you only teach a chef to cook Italian food, they won't be prepared to cook French cuisine

Similarly, if you train an AI model with data that doesn't represent all its future tasks, it won't perform well on those tasks

Data representativeness means ensuring your training data:

Covers all the types of tasks the model will encounter

Includes diverse examples and scenarios

Reflects the real-world distribution of cases

For example:

If you're training a model to handle customer service queries, your training data should include:

Different types of customer issues

Various writing styles

Multiple languages (if applicable)

Different levels of complexity

Both common and edge cases

This is different from:

Data curation (selecting and organizing data)

Data labeling (adding annotations/tags to data)

Data governance (managing data security and compliance)

The correct answer is data representativeness because it specifically focuses on ensuring the training data matches the actual use cases the model will encounter in production.



Question 20
Incorrect
Which technique involves adjusting a model to apply its learned knowledge to new but related tasks?
(A)Instruction tuning
Correct answer
(B)Transfer learning
Your answer is incorrect
(C)Continuous pre-training
(D)Reinforcement learning from human feedback (RLHF)
Overall explanation
Transfer learning involves adapting a model’s learned knowledge from one task to new but related tasks, leveraging the pre-existing capabilities of the model for new applications.

正確答案為**(B) 轉移學習**。
轉移學習涉及調整模型，將其學習到的知識應用於新的但相關的任務。它利用模型預先存在的性能來完成新的應用。這就像一個學生利用他們在一個學科中學到的知識來幫助他們學習另一個相關學科。
其他的選項不正確，原因如下：
●
(A) 指令微調是一種通過提供模型遵循的特定指令來調整模型的技術，但它不一定是針對新任務的。
●
(C) 持續預訓練是指在新的資料上持續訓練一個已經預先訓練好的模型，以進一步提升其性能或使其適應新的領域，但它不特別強調應用於新任務。
●
(D) 人類回饋強化學習(RLHF)**是一種利用人類回饋來訓練強化學習模型的技術，它不一定涉及將學習到的知識應用於新任務。
來源中沒有關於鼓勵模型產生更多創意輸出、提供額外上下文以增進模型理解或根據最少輸入產生回應的資訊。