h = [int(i) for i in input().strip().split(" ")]
word = input().strip()

h = dict(zip([chr(i+97) for i in range(26)], h))

maxH = None
for i in word:
    if maxH == None or h[i] > maxH:
        maxH = h[i]

print(maxH * len(word))



