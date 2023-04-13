"""
https://pub.towardsai.net/chatgpt-api-101-a-beginners-guide-95e6d41c716f

https://aneejian.com/getting-started-chat-gpt-api-comprehensive-guide/
"""


# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

###

import openai
openai.api_key ="blah"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "Hello, how are you?"}]
)

completion['choices'][0]["message"]["content"]

###

prompt = """Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

Context:
Apple quietly included a feature called Clean Energy Charging in iOS 16.1 and turned it on by default. Here's what you need to know about the environmentally conscious feature.
Apple was vocal about Clean Energy Charging when it announced iOS 16 in September 2022. It didn't launch to the public until October 24 with iOS 16.1, and it seemed the change went by without many users noticing - until a recent social media storm.

Sunday, February 26, saw a slew of posts getting attention on Twitter about Clean Energy Charging. Users were vocal and angry, stating that they didn't want Apple to decide how they used energy for them.

The feature is opt-out, so if you don't want to participate in Clean Energy Charging, there is a simple toggle in settings to turn it off.

Q: What is clean energy charging?
A:"""

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": prompt}]
)

completion['choices'][0]["message"]["content"]

###

# Over a large document, like a pdf

import fitz

doc = fitz.open('bert.pdf')
text = ""
for page in doc:
  text += page.get_text()

# tokenize with tiktoken
# break into chunks of 500 tokens each
# get the top k sub-contexts by calculating cosine ismilarity between question and context of each chunk
#     eg with davinci api as below


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    question="Am I a monkey?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    prompt = f"""Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say "I don't know"

    Context:{context}

    Q:{question}
    A:"""
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""




# Prior assistant messages provide context
# If a conversation cannot fit within the model’s token limit, 
# then it will need to be shortened in some way.
# 4096 tokens for gpt-3.5-turbo-0301
# Both input and output tokens count toward the limit
# response['usage']['total_tokens']
# or check with https://github.com/openai/tiktoken library
# as per: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb 



"""
# example response

{
 'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-3.5-turbo',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'},
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}


response['choices'][0]['message']['content']
"""







