def fibonacci_sequence(n):
    """Returns the nth number in the Fibonacci sequence"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a
