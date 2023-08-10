from pynqhls.stream import streamOverlay
overlay = streamOverlay('stream.bit')

signal = range(0, 1000)
coeffs = [1, 0, 0, 0, 0, 0, 0, 0, 0]
output = overlay.run(coeffs, signal)

test = [s == o for (s, o) in zip(signal, output)]

if False in test:
    print("Test Failed!")
else:
    print("Test Passed")

from PIL import Image
rgb = Image.open("pictures/valve.png")
bw = rgb.convert("L")
bw = bw.resize((400,300),Image.ANTIALIAS)
bw

coeffs = [0, 0, -1, -2, 0, 2, 1, 0, 0]
sig = list(bw.getdata())
output = []
for i in range(0, len(sig), 1000):
    output += overlay.run(coeffs, sig[i : i + 1000])

out = Image.new(bw.mode, bw.size)
out.putdata(output)
out



