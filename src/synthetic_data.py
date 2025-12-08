"""
Shared synthetic dataset for CRF and BiLSTM-CRF experiments
"""


def generate_challenging_train_set(num_examples=80):
    templates = [
        (["The", "battery", "is", "bad"], ["O", "B-NEG", "O", "O"]),
        (["Battery", "is", "terrible"], ["B-NEG", "O", "O"]),
        (["The", "screen", "is", "great"], ["O", "B-POS", "O", "O"]),
        (["Screen", "is", "amazing"], ["B-POS", "O", "O"]),
        (["The", "battery", "is", "not", "bad"], ["O", "B-POS", "O", "O", "O"]),
        (["Screen", "is", "not", "terrible"], ["B-POS", "O", "O", "O"]),
        (
            ["The", "keyboard", "quality", "is", "poor"],
            ["O", "B-NEG", "I-NEG", "O", "O"],
        ),
        (["Keyboard", "quality", "is", "excellent"], ["B-POS", "I-POS", "O", "O"]),
    ]

    data = []
    for _ in range(num_examples // len(templates) + 1):
        data.extend(templates)
    return data[:num_examples]


def generate_challenging_test_set():
    return [
        (
            [
                "The",
                "battery",
                "lasts",
                "only",
                "an",
                "hour",
                "which",
                "is",
                "disappointing",
            ],
            ["O", "B-NEG", "O", "O", "O", "O", "O", "O", "O"],
        ),
        (
            ["Battery", "drains", "so", "fast", "it", "is", "frustrating"],
            ["B-NEG", "O", "O", "O", "O", "O", "O"],
        ),
        (["The", "performance", "is", "fast"], ["O", "B-POS", "O", "O"]),
        (["Battery", "drains", "too", "fast"], ["B-NEG", "O", "O", "O"]),
        (
            ["The", "screen", "refresh", "is", "smooth"],
            ["O", "B-POS", "I-POS", "O", "O"],
        ),
        (
            ["The", "typing", "experience", "is", "smooth", "but", "slow"],
            ["O", "B-NEU", "I-NEU", "O", "O", "O", "O"],
        ),
        (
            ["The", "keyboard", "has", "only", "two", "hour", "battery"],
            ["O", "B-NEG", "O", "O", "O", "O", "O"],
        ),
        (
            [
                "I",
                "need",
                "to",
                "charge",
                "the",
                "laptop",
                "three",
                "times",
                "a",
                "day",
            ],
            ["O", "O", "", "O", "O", "B-NEG", "O", "O", "O", "O"],
        ),
        (
            ["The", "fan", "makes", "a", "constant", "noise"],
            ["O", "B-NEG", "O", "O", "O", "O"],
        ),
        (
            ["The", "screen", "is", "not", "as", "bright", "as", "other", "laptops"],
            ["O", "B-NEG", "O", "O", "O", "O", "O", "O", "O"],
        ),
        (
            ["Performance", "is", "not", "what", "I", "expected"],
            ["B-NEG", "O", "O", "O", "O", "O"],
        ),
        (
            [
                "The",
                "keyboard",
                "feels",
                "good",
                "but",
                "the",
                "trackpad",
                "is",
                "awful",
            ],
            ["O", "B-POS", "O", "O", "O", "O", "B-NEG", "O", "O"],
        ),
        (
            ["Battery", "lasts", "all", "day", "however", "the", "screen", "is", "dim"],
            ["B-POS", "O", "O", "O", "O", "O", "B-NEG", "O", "O"],
        ),
        (
            ["Great", "another", "day", "without", "battery"],
            ["O", "O", "O", "O", "B-NEG"],
        ),
        (
            ["Wonderful", "the", "keyboard", "broke", "after", "a", "month"],
            ["O", "O", "B-NEG", "O", "O", "O", "O"],
        ),
    ]
