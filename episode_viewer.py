#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
from torch.utils.data import DataLoader
from os import system
import curses
from collections import OrderedDict
import cv2
from itertools import count

from memory import ExpertDataset
from utils import Arguments


RELEVANT_ITEMS = [
    "log",
    "stick",
    "planks",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "stone_pickaxe",
    "furnace",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe"
]


# define our clear function
def clear_out():
    _ = system('clear')


def get_data_iter(args: Arguments):
    def collate_fn(batch):
        return batch[0]
    data_dir = os.getenv('MINERL_DATA_ROOT', 'data/')
    env = args.env_name
    data_dir = '/'.join([str(data_dir), env])

    expert_dataset = ExpertDataset(data_dir=data_dir, frame_skip=args.frame_skip)
    data = DataLoader(expert_dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)
    return data


def print_data(args: Arguments, screen, idx, inc, meta, states, next_states, actions, rewards, policies):
    obtain_env = 'Obtain' in args.env_name
    c = count(start=0, step=1)
    screen.erase()
    all_actions = args.movement_actions
    all_actions = all_actions + args.crafting_actions if obtain_env else all_actions
    all_actions += ['camera']
    screen.addstr(next(c), 0, f"Step increment: {inc}")
    screen.addstr(next(c), 0, f"Step: {idx}")
    screen.addstr(next(c), 0, f"Next policy: {policies[idx]}")
    screen.addstr(next(c), 0, f"Stream name: {meta['stream_name']}")
    screen.addstr(next(c), 0, f"Reward: {rewards[idx]}")
    curr_action = OrderedDict([
        (key, value[idx]) for key, value in actions.items()
        if key in all_actions
    ])
    for key in all_actions:
        curr_action.move_to_end(key)
    screen.addstr(next(c), 0, f"Action: {curr_action}")

    if obtain_env:
        curr_inv = OrderedDict([
            (key, value[idx]) for key, value in states['inventory'].items() if key in RELEVANT_ITEMS
        ])
        nxt_inv = OrderedDict([
            (key, value[idx]) for key, value in next_states['inventory'].items() if key in RELEVANT_ITEMS
        ])
        for key in RELEVANT_ITEMS:
            curr_inv.move_to_end(key)
            nxt_inv.move_to_end(key)
        screen.addstr(next(c), 0, f"Inventory:      {curr_inv}")
        screen.addstr(next(c), 0, f"Next Inventory: {nxt_inv}")


def show_image(img):
    output = cv2.resize(img, (512, 512))
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame', output)
    cv2.waitKey(1)


if __name__ == '__main__':
    args = Arguments(underscores_to_dashes=True).parse_args(known_only=True)
    data = iter(get_data_iter(args))
    cv2.startWindowThread()
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 600, 600)
    idx = 0
    inc = 1
    states, actions, rewards, next_states, dones, policies, meta, *_ = next(data)
    seq_len = len(rewards)

    screen = curses.initscr()
    curses.noecho()
    curses.cbreak()
    screen.keypad(True)
    # args, screen, idx, inc, meta, states, next_states, actions, rewards, policies
    print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
    show_image(states['pov'][idx])
    try:
        while True:
            char = screen.getch()
            if char == ord('q'):  # Quit
                break
            elif char == ord('x'):  # Speed up
                inc *= 2
                print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
            elif char == ord('z'):  # Slow down
                inc = int(max(1, inc/2))
                print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
            elif char == ord('n'):  # Next observation
                idx = 0
                states, actions, rewards, next_states, dones, policies, meta, *_ = next(data)
                seq_len = len(rewards)
                print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
                show_image(states['pov'][idx])
            elif char == ord('c'):  # Clear screen
                screen.erase()

            elif char == curses.KEY_RIGHT:
                if idx + inc >= seq_len:
                    continue
                idx += inc
                print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
                show_image(states['pov'][idx])
            elif char == curses.KEY_LEFT:
                if idx - inc < 0:
                    continue
                idx -= inc
                print_data(args, screen, idx, inc, meta, states, next_states, actions, rewards, policies)
                show_image(states['pov'][idx])

    except Exception as e:
        print(f"Exception caught: {e}")

    finally:
        # shut down cleanly
        curses.nocbreak()
        screen.keypad(0)
        curses.echo()
        curses.endwin()
        exit()
