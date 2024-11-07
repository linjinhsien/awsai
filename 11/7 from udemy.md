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