# Fym

Very early WIP, putting it on GitHub just because.

Fym = Functional Gym (get it?)

The core idea is turning [gym](https://github.com/openai/gym) into something with
more of a functional approach. It offers the same core functionality, as can be seen in `fym/utils/conversions.py`.
The big conceptual difference is that there is no mutable state - everything is handled through class methods.

The disadvantage of this approach is that for some environments, this simply won't work - think Atari or Unity3d. 

The advantage is the immutability and a better connection to actual MDP-like components. 
And at least some frameworks should support this, like [brax](https://github.com/google/brax). 

Note: the RLE notation stands for Reinforcement Learning Environment. 
It's connected to a paper that I might submit sometime this month. Don't worry about it for now.