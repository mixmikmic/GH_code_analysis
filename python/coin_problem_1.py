# Test procedure for each trial:
#   * Choose a coin (10% chance of unfair, 90% chance of fair)
#   * Flip that coin 5 times; if any flip is not heads, reject the sample and go to next trial
#   * If all 5 flips are heads, count this sample, and flip the coin a last time, if heads, count as success

import random

p_f = 0.9
p_h_f = 0.5
trials = 1000000

def fair_coin():
    return random.random() <= 0.5

def unfair_coin():
    return True

def get_coin():
    if random.random() <= p_f:
        return fair_coin
    else:
        return unfair_coin

got_5_heads = 0
got_6th_flip = 0
for i in range(trials):
    coin_got_heads = get_coin()

    reject_sample = False

    for flip in range(5):
        if not coin_got_heads():
            # This coin didn't get 5 heads in a row, so reject it
            reject_sample = True
            break

    if reject_sample:
        continue

    got_5_heads += 1

    if coin_got_heads():
        got_6th_flip += 1

prob = got_6th_flip / got_5_heads
print('P(h6|h1, h2, h3, h4, h5) = {}'.format(prob))



