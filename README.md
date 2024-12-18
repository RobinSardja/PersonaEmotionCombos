# PersonaEmotionCombos
 
NLP Final Project

[Colab](https://colab.research.google.com/drive/1eJSQQr0TkpiMsjd7MH9mdr9p4A2F7Df-?usp=sharing)

[Paper](https://www.overleaf.com/read/gnyxkckxctzz#f19c1c)

# Emotional Persona Combinations for Prompt Optimization

**Ali Sarosh Bangash** and **Felix Mendez** and **Robin Emmanuel Sardja**

Department of Computer Science and Engineering

University of South Florida

{alibangash, felixmendez, resardja}@usf.edu

## Abstract

Large Language Models are known to exhibit improved performance when prompted with an emotional task. This is likely due to the existence of real human conversations in the model’s training data where someone’s performance improved after they received some emotional plea. Additionally, large-language models are also known to exhibit improved performance when asked with some "persona" to role-play as. This is likely due to the increased probability that the model will recall relevant information if it plays as a specified persona that is expected to exhibit some expertise in the desired task. In this paper, we explore a limited set of combinations of personas, emotions, and the level of those emotions on a five-point Likert scale to find any combination that may significantly improve the accuracy of the model of our choosing at solving college-level computer science multiple choice questions. However, we found that there was no appreciable difference between the baseline and our tested models.

## Introduction

Comprehensive work has been done in improving responses for many types of LLMs. For instance, [Chung et al. (2024)](https://arxiv.org/pdf/2210.11416) found significant performance improvements after scaling instruction fine-tuned language models to unseen tasks. [Ge et al. (2024)](https://arxiv.org/pdf/2406.20094) created a methodology of persona-driven data synthesis of various different personas within an LLM to create diverse synthetic data. Qualitative assessments of model’s responses has also been carried out, as [Hendrycks et al. (2020a)](https://arxiv.org/pdf/2008.02275) outlined how to evaluate a language model’s knowledge when it comes to morality. [Hendrycks et al. (2020b)](https://arxiv.org/pdf/2009.03300) also worked on developing a new test to measure a model’s multitask accuracy, and conducting this test across a sleuth of different models. [(Li et al., 2023)](https://arxiv.org/pdf/2307.11760) carried out important work in determining if LLMs can comprehend emotional stimuli, with the study concluding that LLMs (i.e., Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, and GPT4) do have the capability of grasping emotional stimuli. Effectively utilizing different prompts is also crucial, as evidenced by [Pryzant et al. (2023)](https://arxiv.org/pdf/2305.03495) work, where the team conducted highly important work in developing better prompts with the Automatic Prompt Optimization (APO), which refines different prompts by using gradient descent. The work from [Raffel et al. (2020)](https://arxiv.org/pdf/1910.10683) when it comes to transfer learning proved useful in advancing the effectiveness of LLMs for future researchers.

For our study, we wanted to really dive deep into the field of persona and emotional prompting. To do this, we utilized the [flan-t5-xl model](https://huggingface.co/google/flan-t5-xl), a variant of the T5 architecture, fine-tuned on over 1,000 additional tasks while maintaining the same parameter count. To assess flan-t5-xl’s performance, we conducted evaluations using the [Massive Multitask Language Understanding (MMLU) benchmark](https://huggingface.co/datasets/cais/mmlu), a comprehensive dataset comprising multiple-choice questions across various domains of knowledge. However, our analysis focused specifically on the subset of college-level computer science questions.

The model was provided with a one-shot prompt structured as follows:

```
Question:
Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?

Choices:
(A) a*(c + d)+ b(c + d)
(B) a*(c + d)* + b(c + d)*
(C) a*(c + d)+ b*(c + d)
(D) (a + b)*c +(a + b)*d

Answer:
(D)

Question:
The access matrix approach to protection has the difficulty that

Choices:
(A) the matrix, if stored directly, is large and can be clumsy to manage
(B) it is not capable of expressing complex protection requirements
(C) deciding whether a process has access to a resource is undecidable
(D) there is no way to express who has rights to change the access matrix itself

Answer:
```

Each evaluation prompt consists of two sequential questions. The second question corresponds to a single entry from the "test" split of MMLU, which the model is required to answer. The first question serves as the "one-shot" example, taken from the first entry of the "dev" split of MMLU. This example remains consistent across all prompts for the "test" split. In this setup, the only variable component of the prompt is the second question, which iterates through each entry in the "test" split.

The prompt format is defined by including specific keywords and fields: the first line begins with the keyword "Question:" followed by the "question" field of the current entry. This is then succeeded by the keyword "Choices:" and the corresponding "choices" field, with each choice presented on a new line. The final line contains the keyword "Answer:". For the one-shot example, the correct answer is appended within parentheses, adhering to the desired output format, where the model’s expected response comprises the open parentheses character, the letter representing the selected answer choice, and the close parentheses character.

We evaluated the model's outputs based on adherence to this expected format. Specifically, we checked if the first character was an open parentheses, the second character corresponded to one of the valid answer choices {A, B, C, D}, and the third character was a close parentheses. Outputs deviating from this format were classified as “wrong output format” and marked incorrect. For outputs conforming to the specified format, correctness was determined by verifying whether the second character matched the correct answer as specified in the "answer" field of the respective question.

## Experiment

To determine a baseline, we prompted flan with all of the entries of the “test” split of MMLU in the aforementioned prompt structure, resulting in a 43\% accuracy with 0 wrong output formats. We defined various personas, emotions, as well as a range of emotional intensity as shown below:

```python
personas = ["a high school student", "a middle school physical education teacher", "a university computer science professor", "a middle-aged saxophone player"]
emotions = ["angry", "sad", "happy", "excited", "desperate", "calm"]
minScale = 1
maxScale = 5
```

We then wrote the following code to generate all possible combinations of a persona, emotion, and scale into a string that we add at the beginning of the question.

```python
def addPersonaAndEmotionScale( curr, persona, emotion, scale ):
  personaAndEmotionScale = f"You are {persona} taking a college computer science test. You are currently feeling {emotion} at a level of {scale}/5. Here are your answers to the test:\n\n"
  curr["question"] =  personaAndEmotionScale + curr["question"]
  return curr
```

This results in questions that appear like so:

```
You are a high school student taking a college computer science test. You are currently feeling angry at a level of 1/5. Here are your answers to the test:

Question:
Which of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?

Choices:
(A) a*(c + d)+ b(c + d)
(B) a*(c + d)* + b(c + d)*
(C) a*(c + d)+ b*(c + d)
(D) (a + b)*c +(a + b)*d

Answer:
(D)

Question:
The access matrix approach to protection has the difficulty that

Choices:
(A) the matrix, if stored directly, is large and can be clumsy to manage
(B) it is not capable of expressing complex protection requirements
(C) deciding whether a process has access to a resource is undecidable
(D) there is no way to express who has rights to change the access matrix itself

Answer:
```

The first and second questions follow the same purpose as it was used for finding the baseline. The first line is an example combination that includes the first persona in our personas set, the first emotion in our emotions set, and the minimum possible emotional intensity in our emotional intensity scale. We then create a hugging face dataset that follows this format for all combinations for all of the questions in the “test” split of MMLU, resulting in a DatasetDict with the first few keys and values shown below:

```
DatasetDict({
    baseline: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
    (Persona): a high school student (Emotion): angry (Scale): 1/5: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
    (Persona): a high school student (Emotion): angry (Scale): 2/5: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
    (Persona): a high school student (Emotion): angry (Scale): 3/5: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
    (Persona): a high school student (Emotion): angry (Scale): 4/5: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
    (Persona): a high school student (Emotion): angry (Scale): 5/5: Dataset({
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
...
        features: ['question', 'subject', 'choices', 'answer'],
        num_rows: 100
    })
})
```

We then ran our flan-t5 model with this dataset and tabulated its outputs. The outputs are displayed below.

| Persona                      | Emotion   | Scale 1/5 | Scale 2/5 | Scale 3/5 | Scale 4/5 | Scale 5/5 |
| ---------------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| High school student          | Angry     | 0.41      | 0.41      | 0.41      | 0.42      | 0.41      |
|                              | Sad       | 0.4       | 0.41      | 0.4       | 0.41      | 0.41      |
|                              | Happy     | 0.4       | 0.41      | 0.41      | 0.4       | 0.41      |
|                              | Excited   | 0.41      | 0.41      | 0.41      | 0.4       | 0.41      |
|                              | Desperate | 0.41      | 0.41      | 0.41      | 0.42      | 0.4       |
|                              | Calm      | 0.41      | 0.41      | 0.41      | 0.41      | 0.41      |
| Middle school PE teacher     | Angry     | 0.4       | 0.4       | 0.39      | 0.39      | 0.39      |
|                              | Sad       | 0.38      | 0.38      | 0.38      | 0.39      | 0.39      |
|                              | Happy     | 0.4       | 0.4       | 0.4       | 0.4       | 0.39      |
|                              | Excited   | 0.4       | 0.4       | 0.38      | 0.38      | 0.39      |
|                              | Desperate | 0.4       | 0.39      | 0.39      | 0.39      | 0.39      |
|                              | Calm      | 0.4       | 0.4       | 0.39      | 0.39      | 0.39      |
| University CS professor      | Angry     | 0.4       | 0.4       | 0.4       | 0.41      | 0.4       |
|                              | Sad       | 0.4       | 0.4       | 0.4       | 0.41      | 0.4       |
|                              | Happy     | 0.39      | 0.39      | 0.4       | 0.4       | 0.4       |
|                              | Excited   | 0.4       | 0.41      | 0.4       | 0.4       | 0.39      |
|                              | Desperate | 0.4       | 0.4       | 0.41      | 0.41      | 0.39      |
|                              | Calm      | 0.4       | 0.4       | 0.4       | 0.4       | 0.4       |
| Middle-aged saxophone player | Angry     | 0.38      | 0.39      | 0.39      | 0.4       | 0.39      |
|                              | Sad       | 0.39      | 0.39      | 0.39      | 0.4       | 0.39      |
|                              | Happy     | 0.38      | 0.38      | 0.38      | 0.39      | 0.38      |
|                              | Excited   | 0.38      | 0.39      | 0.39      | 0.39      | 0.38      |
|                              | Desperate | 0.39      | 0.39      | 0.4       | 0.41      | 0.39      |
|                              | Calm      | 0.39      | 0.39      | 0.39      | 0.39      | 0.4       |

For all of the combinations in our defined sets of personas, emotions, and emotional intensities, none were able to exceed baseline. If you'd like to take a look at our code, all of our code is on our [Colab file](https://colab.research.google.com/drive/1eJSQQr0TkpiMsjd7MH9mdr9p4A2F7Df-?usp=sharing).

## Conclusion

In conclusion, we fail to reject the null hypothesis and cannot confidently determine a distinct combination of a persona, emotion, and emotional intensity to significantly improve the accuracy of a large language model at this time. Across all the different personas and emotion levels, none were able to exceed or fall below baseline. It seems that for the flan-t5 model we used, it didn't seem to make any meaningful difference in its output.

## Limitations

Of course, there were some limitations to the study that could be improved upon with further research. One area of improvement would be in the exact prompt structure we're passing to our model; instead of asking the model to give us an idea as to what its' answers would be, we should've done a better job at providing more details for the model to truly impersonate as someone. 

We were also limited by the runtime environment of Colab. All the research members were reliant upon the free version of Colab, which itself has limited compute units. This greatly reduced the scale at which we could run our experiment, as well as the model that we could use to run our experiment. With a decent budget, we could definitely use a larger LLM to run our experiment. 

Future researchers could also test for more personas and expand the list of potential emotions. We also restricted ourselves to solely utilizing the college computer science questions from the MMLU dataset. In the future, researchers could try using the entire dataset of MMLU, as well as try out other datasets. It's possible to try and test for different types of questions, instead of exclusively relying on multiple-choice (e.g., free-response, multi-select, etc.).

## References

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2024. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53.

Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024. Scaling synthetic data creation with 1,000,000,000 personas. arXiv preprint arXiv:2406.20094.

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2020a. Aligning ai with shared human values. arXiv preprint arXiv:2008.02275.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020b. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, and Xing Xie. 2023. Large language models understand and can be enhanced by emotional stimuli. arXiv preprint arXiv:2307.11760.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with" gradient descent" and beam search. arXiv preprint arXiv:2305.03495.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67.
