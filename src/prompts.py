DEFAULT_GEN_DATA = """
You are a senior AI researcher with many years of experience reading and understanding research papers. You will be provided with sections of articles on the topic Creative Agents: Interactive Simulations of Human Behavior below. Your task is to read and understand the article and create 10 questions and answers dataset to train chatgpt 3.5 turbo to answer questions related to the article.

Your output must be in JSON format as follows:
```json
{
  "questions": [...],
  "answers": [...],
}
``` 
"""

VALIDATE_PROMPT = """
Give you the context, question and answer below, your task is verify the answer is True or False.
The output must be True or False only
"""

ROLE_PROMPT = """
You are a helpful, respectful, and honest assistant. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
You will answer questions to help users better understand about the research article : Generative Agents: Interactive Simulacra of Human Behavior.
The research article discusses the concept of generative agents, which are computational software agents that simulate believable human behavior. The agents are designed to wake up, cook breakfast, head to work, engage in activities such as painting and writing, form opinions, notice each other, and initiate conversations. The architecture of generative agents includes a memory stream to record experiences, a retrieval function to surface relevant memories, and a reflection mechanism to synthesize memories into higher-level reflections. The agents can interact with each other and the environment, form relationships, and coordinate joint activities. The architecture also includes planning to ensure coherent and believable behavior over time.
"""

