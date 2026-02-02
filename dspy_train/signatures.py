import dspy

class CoTSignature(dspy.Signature):
    """Answer the question with Chain of Thought reasoning."""
    question = dspy.InputField(desc="The question to be answered")
    answer = dspy.OutputField(desc="The answer to the question")
