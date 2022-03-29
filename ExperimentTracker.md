# Experiment tracker

Central note taking sheet for experiments. Used with [wandb.](https://wandb.ai/antoniolopardo/PracticalWork?workspace=user-antoniolopardo)

## AsDiv

### baseline_asdiv

**Description**: Insert "write a program that writes" between the context and the question, 15 examples included in the prompt.

**Result**: `0.4072164948453608` - [wandb](https://wandb.ai/antoniolopardo/PracticalWork/runs/20grqy5o?workspace=user-antoniolopardo)

*# Each house a carpenter builds needs six sinks. If he bought two hundred sixty-six sinks, write a program that writes How many houses would that cover? {python_program}*

### prefix_asdiv

**Description**: Prefix "Write a program that prints the answer to the following question." at the begining of the quesiton, 6 examples included.

**Result**: `TODO`

## GSM8k

### baseline_gsm8k

**Description**: Prefix "Write a program that prints the answer to the following question." at the begining of the the quesiton, 6 examples included. Leaving a couple of empy lines at the end seems to help.

**Result**: `0.065` - [wandb](https://wandb.ai/antoniolopardo/PracticalWork/runs/1d6scg9p?workspace=user-antoniolopardo)

*"# Write a program that prints the answer to the following question. To run his grocery store, Mr. Haj needs $4000 a day. This money is used to pay for orders done, delivery costs and employees' salaries. If he spends 2/5 of the total operation costs on employees' salary and 1/4 of the remaining amount on delivery costs, how much money he pays for the orders done? {python_program}"*

### only_equations_gsm8k

**Description**: Prefix "Write a program that executes the following equations and prints the result of the last one" then on ly include the equations, nothing else.

**Result**: `TODO`