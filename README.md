# Deep Reinfocement Learning for Stocks

This repository intends to leverage the power of Deep Reinforcement Learning for the Stock Market. The algorithm is based on Xiong et al Practical Deep Learning Approach for Stock Trading. However, instead of using the traditional DDPG algorithm, we use Twin-Delayed DDPG. Additionally, we constructured our system to be able to trade multiple stocks at once, instead of the "one stock at a time" approach that they adapted in their paper

## Instructions

Simply run:

```python
python main.py
```

And the algorithm will proceed to run with the stocks listed in stock_names (in main.py)

## Contributing

Daniel Chang, Austin Starks, Justin Starks, and Ryan Oh contributed to this repository.

## References

Fujimoto, S., Van Hoof, H., & Meger, D. (2018, October 22). Addressing Function Approximation Error in Actor-Critic Methods. arxiv:1802.09477

Xiong, Z., Liu, X. Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement
learning approach for stock trading. arXiv preprint arXiv:1811.07522.
