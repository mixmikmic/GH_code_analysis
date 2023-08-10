def neg_bernoulli(n, p):
    return p * (1-p)**(n-1)

defective_prob = 1 / 3.0
inspection = 5

round(neg_bernoulli(inspection, defective_prob), 3)

neg_bernoulli(12/100.0, 100)



