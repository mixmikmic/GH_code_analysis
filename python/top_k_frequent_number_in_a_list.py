def top_k_frequent_number(nums,k):
    num_map = {1:0, 2:0,3:0,4:0,5:0,6:0}

    for num in nums:
        num_map[num] += 1

    ret = []
    while k > 0:
        max_freq = 0
        max_key = 0
        for key in num_map:
            if num_map[key] > max_freq:
                max_freq = num_map[key]
                max_key = key
        ret.append(max_key)
        del num_map[max_key]
        k -= 1
    return ret

print(top_k_frequent_number([3,2,3,1,2,5,3,6,2,4],2))

print(top_k_frequent_number([1,1,1,2,2,3,3,4,4],3))

