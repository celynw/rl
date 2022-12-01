#!/usr/bin/env python3
# import colored_traceback.auto
import colored_traceback.always

# from rich import print as print_
# import sys
# print_(f"TTY: {sys.stdout.isatty()}")

# import colorama
# print(f"{colorama.Fore.RED}Red{colorama.Fore.RESET}")

# import pygments
# import pygments.lexers
# from pygments.formatters import get_formatter_by_name
# fmt_options = {"style": "default"}
# formatter = get_formatter_by_name("terminal256", **fmt_options)
# lexer = pygments.lexers.get_lexer_by_name("pytb", stripall=True)
# error = """
# Traceback (most recent call last):
#   File "/home/cw0071/dev/python/rl/./ctb.py", line 21, in <module>
#     print(a)
# NameError: name 'a' is not defined"""
# print(pygments.highlight(error, lexer, formatter))

# import curses
import os
print(f"TERM: {os.getenv('TERM')}")
# curses.setupterm("xterm-256color")

print(a)
