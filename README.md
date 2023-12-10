# Deep Reinfocement Learning for Stocks

Full Paper Here: https://drive.google.com/file/d/1x67IaLpErVw9SwSBjWAdDtNEOcQSgje_/view?usp=sharing

This repository intends to leverage the power of Deep Reinforcement Learning for the Stock Market. The algorithm is based on Xiong et al Practical Deep Learning Approach for Stock Trading. However, instead of using the traditional DDPG algorithm, we use Twin-Delayed DDPG. Additionally, we constructed our system to be able to trade multiple stocks at once, instead of the "one stock at a time" approach that they adapted in their paper

## Instructions

Simply clone the repository

```python
git clone https://github.com/austin-starks/Deep-RL-Stocks.git
```

And run:

```python
python main.py
```

And the algorithm will proceed to run with the stocks listed in stock_names (in main.py).

Currently, this system supports every stock in the S&P 500 and the Nasdaq. If you want to train/test this algorithm with stock(s) that are not in these indices, you can download the data using the "collect_data.ipynb" file and run it on the stock(s) that you desire

## Contributing

Daniel Chang, Austin Starks, Justin Starks, and Ryan Oh contributed to this repository.

## References

Fujimoto, S., Van Hoof, H., & Meger, D. (2018, October 22). Addressing Function Approximation Error in Actor-Critic Methods. arxiv:1802.09477

Xiong, Z., Liu, X. Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement
learning approach for stock trading. arXiv preprint arXiv:1811.07522.
