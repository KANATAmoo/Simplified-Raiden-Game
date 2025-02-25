# Simplified-Raiden-Game

**1.Game Design**

This game is a Q-learning-based AI training program for a Raiden-style game. The player controls an aircraft with the goal of dodging enemies and bullets to survive as long as possible. The game uses simple 2D graphics, with enemies and bullets represented by colored squares. The player's aircraft is a blue square. The core of the game is to train the AI to master dodging strategies through reinforcement learning.

**2.Q-learning Implementation**

Q-learning is a reinforcement learning algorithm suitable for discrete state and action spaces. In this game, the state space includes the player's position and the direction of the nearest bullet. The action space includes staying still, moving left, right, up, or down.

The reward function:
- Surviving each frame: +1
- Colliding with an enemy or bullet: -1000
- Moving away from bullets: +0.1 * distance

The Q-value update formula:
Q (s, a) = (1− α) Q (s, a ) +α (r +γ maxQ (s′,a′))
where α is the learning rate and γ is the discount factor.

**3.Evaluation Results**

After training, the AI's performance improved significantly:
- Early episodes: Average survival time of about 10 seconds, frequent collisions.
- Mid-term episodes: Average survival time increased to 30 seconds, beginning to dodge simple bullet patterns.
- Late episodes: Average survival time exceeded 60 seconds, efficiently dodging complex bullet patterns.

**4.Challenges and Solutions**

State Space Explosion:
- Challenge: The original state space was too large, leading to low learning efficiency.
- Solution: Discretize the player's position into grids and track only the direction of the nearest bullet.

Exploration vs. Exploitation Balance:
- Challenge: Fixed exploration rate caused reduced learning efficiency in later stages.
- Solution: Introduce decaying exploration rate.

Sparse Rewards:
- Challenge: Relying solely on survival rewards led to slow learning.
- Solution: Add distance rewards to encourage the AI to move away from threats.

**5.Conclusion**

This game demonstrates the effectiveness of Q-learning in a simple game environment. Through reasonable state representation and reward design, the AI can learn complex dodging strategies. Future work includes introducing more enemy types, optimizing state representation, and exploring more advanced reinforcement learning algorithms.

**Vedio Link: https://youtu.be/SpP7NfWNkJs**

When recording a video, I was unable to make the screen full screen due to a problem with the recording software. This resulted in the cursor position not matching the actual screen display position.
