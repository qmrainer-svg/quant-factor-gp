# quant-factor-gp

> A research framework for discovering and evaluating quantitative alpha factors using Genetic Programming.

---

## 📌 Overview

`quant-factor-gp` is an experimental research framework for **automatic quantitative factor (alpha) discovery** based on **Genetic Programming (GP)**.

The goal of this project is to:

- Automatically generate candidate alpha factors using symbolic expression trees
- Evaluate factor quality using cross-sectional backtesting / IC / rank IC
- Evolve factor formulas through genetic operators (mutation, crossover, selection)
- Build a reusable research pipeline for factor mining and experimentation

This project is intended for:

- Quantitative research
- Factor mining experiments
- Research on automated feature / signal discovery
- Exploratory research in alpha generation

---

## 🧠 Core Idea

We represent each alpha factor as a **symbolic expression tree** composed of:

- Basic arithmetic operators: `+ - * /`
- Financial operators: `rank, delay, delta, mean, std, corr, ...`
- Input features: price, volume, returns, etc.

Then we use:

- Genetic Programming to evolve the population
- Fitness function = factor performance metrics (IC, IR, etc.)
- Selection + mutation + crossover to iteratively improve factor quality

