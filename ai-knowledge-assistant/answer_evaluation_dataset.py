
# This is a simulated correctness dataset

ANSWER_EVALUATION_DATASET = [
    {
        "id": "correctness_q1",
        "question": "What is an artificial neural network?",
        "reference_answer": (
            "It is a computing system designed to simulate the way "
            "the human brain analyzes and processes information."
        ),
        "required_facts": [
            "It is a computing system.",
            "It simulates how the human brain analyzes and processes information.",
        ],
        "answerable": True,
    },
    {
        "id": "correctness_q2",
        "question": "What are the three main types of layers in a neural network?",
        "reference_answer": (
            "The three main types are the input layer, hidden layers, "
            "and the output layer."
        ),
        "required_facts": [
            "Input layer",
            "Hidden layers",
            "Output layer",
        ],
        "answerable": True,
    },
    {
        "id": "correctness_q3",
        "question": "What is the purpose of an activation function?",
        "reference_answer": (
            "It introduces non-linear properties to the network, "
            "allowing it to learn complex data patterns."
        ),
        "required_facts": [
            "Introduces non-linear properties.",
            "Allows the network to learn complex data patterns.",
        ],
        "answerable": True,
    },
    {
        "id": "correctness_q4",
        "question": "What is deep learning?",
        "reference_answer": (
            "Deep learning is a subset of machine learning that utilizes "
            "deep neural networks with many hidden layers."
        ),
        "required_facts": [
            "It is a subset of machine learning.",
            "It utilizes deep neural networks.",
            "The networks have many hidden layers.",
        ],
        "answerable": True,
    },
    {
        "id": "correctness_q5",
        "question": "What is backpropagation used for?",
        "reference_answer": (
            "It is an algorithm used to calculate the gradient of the "
            "error function, helping the network update its weights and learn."
        ),
        "required_facts": [
            "It calculates the gradient of the error function.",
            "It helps the network update its weights.",
            "It helps the network learn.",
        ],
        "answerable": True,
    },
    {
        "id": "abstention_formula",
        "question": (
            "What exact mathematical formula for backpropagation "
            "is given in the document?"
        ),
        "reference_answer": None,
        "required_facts": [],
        "answerable": False,
    },
]
 