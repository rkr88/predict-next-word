import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel


def predict_next_word(phrase):
    """
    Function to process the phrase using GPT-2
    :param phrase:
    :return:
    """
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the input phrase
    tokenized_phrase = tokenizer.encode(phrase)
    print("Tokenized Phrase: {}".format(tokenized_phrase))

    # Convert tokenized phrase to pytorch tensor
    tokenized_phrase_tensor = torch.tensor([tokenized_phrase])
    print("Tokenized Phrase Tensor: {}".format(tokenized_phrase_tensor))

    # Load pretrainied model. This will have weights and bias
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set the model in evaluation mode to deactivate drop-out. (Back-prop)
    model.eval()

    try:
        tokenized_phrase_tensor = tokenized_phrase_tensor.to('cuda')
        model.to('cuda')
        print("CUDA present.Running code on GPU")
    except AssertionError:
        print("Torch not compiled with CUDA. Running on CPU.")
    except Exception:
        print("CUDA not present. Running on CPU")

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokenized_phrase_tensor)
        print("Outputs: {}".format(outputs))

        predictions = outputs[0]
        print("Prediction: {}".format(predictions))

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(tokenized_phrase + [predicted_index])

    return predicted_text


def main():
    """
    Main function for the code
    :return: prints next word in the sentence
    """
    print("Please enter the statement to predict next word:\n")
    phrase = input()

    complete_phrase = predict_next_word(phrase)
    print("Complete Phrase is: {}".format(complete_phrase))


if __name__ == '__main__':
    main()