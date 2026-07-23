from macro.main import click, time, ahk
import random


IDK_VARIABLE_DEBOUNCE = 0.05

def a():
    color = (random.choice((0, 255)), random.choice((0, 255)), random.randint(0, 255))
    print(color)
    for i in range(3):
        if color[i] < 10 or color[i] > 245:
            click(151, 677+61*i, 'Down')
            time.sleep(IDK_VARIABLE_DEBOUNCE)
            click(71+162*color[i]/255.0, 677+61*i, 'Up')
        else:
            click(71+162*color[i]/255.0, 677+61*i)


import keyboard

keyboard.add_hotkey('f6', a)
keyboard.wait()