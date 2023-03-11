# read all text into a string

# vocab size: [colors] + special tokens
# should be like 10 + 5

# tokenize the input text
# already have tokens!
# tokenization trade off:
# shorter vocab -> longer sequences
# larger vocab -> shorter sequences
# "hi there" char level encoding -> 8 tokens
# "hi there" gpt2 encoding, 50k vocab size -> 3 tokens