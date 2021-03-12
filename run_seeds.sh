#!/usr/bin/env bash
env_name="$1"
demo_exp_name="$2-demo"
agent_exp_name="$2-agent"

python main.py --local-log --experiment-name "$demo_exp_name" --env-name "$env_name"
python main.py --local-log --seed 432 --experiment-name "$demo_exp_name" --env-name "$env_name"
python main.py --local-log --seed 7019 --experiment-name "$demo_exp_name" --env-name "$env_name"
python main.py --local-log --seed 68472 --experiment-name "$demo_exp_name" --env-name "$env_name"
python main.py --local-log --seed 159375 --experiment-name "$demo_exp_name" --env-name "$env_name"

python main.py --local-log --experiment-name "$agent_exp_name" --env-name "$env_name" --no-expert-memory
python main.py --local-log --seed 432 --experiment-name "$agent_exp_name" --env-name "$env_name" --no-expert-memory
python main.py --local-log --seed 7019 --experiment-name "$agent_exp_name" --env-name "$env_name" --no-expert-memory
python main.py --local-log --seed 68472 --experiment-name "$agent_exp_name" --env-name "$env_name" --no-expert-memory
python main.py --local-log --seed 159375 --experiment-name "$agent_exp_name" --env-name "$env_name" --no-expert-memory