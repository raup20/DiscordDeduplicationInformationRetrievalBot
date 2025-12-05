from hashlib import md5

def simhash(text):
    bits = 64
    v = [0] * bits
    for word in text.lower().split():
        h = int(md5(word.encode()).hexdigest(), 16)
        for i in range(bits):
            bit = (h >> i) & 1
            v[i] += 1 if bit == 1 else -1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint

def hamming_distance(x, y):
    return bin(x ^ y).count("1")
