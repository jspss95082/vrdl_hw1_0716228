---
title: 'VRDL HW1 0716228'
disqus: hackmd
---

VRDL HW1 0716228
===


[![hackmd-github-sync-badge](https://hackmd.io/6p7VjUESSYeoBbkQPXsG3Q/badge)](https://hackmd.io/6p7VjUESSYeoBbkQPXsG3Q)
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## Table of Contents

[TOC]

## How to generate answer.txt?

1. Download [model](https://drive.google.com/file/d/1pno2ta7QvyBvG-jDu00A5czOXxuXxDSr/view?usp=sharing) and [testing_image](https://drive.google.com/drive/folders/1Pea5XbB9RwnOk02V9UNh9UIZwVOatUwT?usp=sharing) from the given link
2. `pip install -r requirements.txt`
3. `python inference.py`

## Train the model
1. Download [training_image](https://drive.google.com/drive/folders/1Pea5XbB9RwnOk02V9UNh9UIZwVOatUwT?usp=sharing) from the given link
2. `python highest.py --epochs 30 --batch-size 10 --seed 7`

###### tags: `VRDL` `Resnext101-32*8d`
