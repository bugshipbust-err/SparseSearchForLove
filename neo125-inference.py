from transformers import pipeline

text = input("prompt: ")

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
out = generator(text, do_sample=True, max_new_tokens=25)

print(out[0]["generated_text"])
