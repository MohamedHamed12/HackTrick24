def split_massage_chunks(msg: str, parts: int) -> list[str]:
    length = len(msg)
    part_size = length // parts
    remainder = length % parts

    splitted_string = []
    start = 0

    for i in range(parts):
        if i < remainder:
            end = start + part_size + 1
        else:
            end = start + part_size

        splitted_string.append(msg[start:end])
        start = end

    return splitted_string


def make_random_massage(real_massages, fake_massages):
    return real_massages+fake_massages
