def split_massage_chunks(msg,num):
    chunks = [msg[i:i+num] for i in range(0, len(msg), num)]
    return chunks

# print(split_massage_chunks("01234567890123456789"))
