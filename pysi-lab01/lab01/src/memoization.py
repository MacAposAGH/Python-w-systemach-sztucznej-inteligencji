memory = [0, 1]

def fib(n: int) -> int:
    if n < 2:
        return n

    memory.append(fib(n - 1) + fib(n - 2))

    return memory[-1]

