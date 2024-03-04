def split_massage_chunks(msg: str, num: int) -> list[str]:
    if len(msg) == 0:
        return ['']
    l = (len(msg) + num - 1) // num
    chunks = [msg[i:i+l] for i in range(0, len(msg), l)]
    return chunks


def make_random_massage(real_massages, fake_massages):
    return real_massages+fake_massages
