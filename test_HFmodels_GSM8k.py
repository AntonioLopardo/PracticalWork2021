from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").eval().cuda()

text = '''
"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
clps_may = 48
clps_april = 48 / 2
total_clip_sold = clps_may + clps_april
print(total_clip_sold)
print(total_clip_sold)

"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?"
budget = 200
spent_on_shirt = 30
spent_on_pants = 46
spent_on_coat = 38
spent_on_socks = 11
spent_on_belt = 18
left_on_budget = 16
spent_on_shoes = budget - (spent_on_shirt + spent_on_pants + spent_on_coat + spent_on_socks + spent_on_belt + left_on_budget)
print(spent_on_shoes)
print(spent_on_shoes)

"question": "Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?"
cheese = 10
cream_cheese = 5
cold_cuts = 20
cost_of_cheese = cheese + cream_cheese
cost_of_cold_cuts = cold_cuts + cost_of_cheese
print(cost_of_cold_cuts)
print(cost_of_cold_cuts)

"question": "The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?"'''

tokens = tokenizer(text, return_tensors="pt").input_ids
generated_tokens = model.generate(tokens.long().cuda(), use_cache=True, do_sample=True, top_k=50, temperature=0.4, top_p=0.9, min_length=1, max_length=len(tokens[0]) + 100, pad_token_id=tokenizer.eos_token_id)
last_tokens = generated_tokens[0][len(tokens[0]):]
generated_text = tokenizer.decode(last_tokens)
print("Generation:\n" + generated_text)