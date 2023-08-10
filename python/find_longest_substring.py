given_string = 'ccaababbaccbabbcc'

def find_longest_substring_distinct(s):
    """
    Find the longest substring made up of only 2 distint chars in s.
    """
    # max_length holds a tuple (max_length, start_idx, stop_idx)
    max_length = (0,0,0)
    left_idx = 0
    right_idx = 0
    # dictionary to hold the character count between the left and the right index
    char_dict = {s[0]:1}
    # iterate over string and update the right index
    while right_idx < len(s)-1:
        if len(char_dict) <= 2:
            # update max length if the current substring is longer than the
            # current maxlength substring
            new_length = right_idx - left_idx + 1
            if new_length > max_length[0]:
                max_length = (new_length, left_idx, right_idx)
            # update the right index and the char_dict
            right_idx += 1
            right_char = s[right_idx]
            if right_char in char_dict:
                char_dict[right_char] += 1
            else:
                char_dict[right_char] = 1
        else:  # len(char_dict) > 2
            # start updating the left index
            left_char = s[left_idx]
            char_dict[left_char] -= 1
            if char_dict[left_char] == 0:
                del char_dict[left_char]
            left_idx += 1
#         print(left_idx, right_idx, s[left_idx:right_idx+1], len(char_dict), char_dict)
    return s[max_length[1]:max_length[2]+1], max_length[0], max_length[1], max_length[2]

print(find_longest_substring_distinct(given_string))

