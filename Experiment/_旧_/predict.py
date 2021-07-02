import sys
import tensorflow as tf


def print_my_examples(inputs, results):
  result_for_printing = \
    [f'input: {inputs[i]:<30} : score: {results[i][0]:.6f}'
                         for i in range(len(inputs))]
  print(*result_for_printing, sep='\n')
  print()


def main(model_path, test_texts):
    reloaded_model = tf.saved_model.load(model_path)

    reloaded_results = tf.sigmoid(reloaded_model(tf.constant(test_texts)))
    original_results = tf.sigmoid(classifier_model(tf.constant(test_texts)))

    print('Results from the saved model:')
    print_my_examples(test_texts, reloaded_results)
    print('Results from the model in memory:')
    print_my_examples(test_texts, original_results)



if __name__ == "__main__":
    saved_model_path = sys.argv[1]

    test_texts = [
        'this is such an amazing movie!',  # this is the same sentence tried earlier
        'The movie was great!',
        'The movie was meh.',
        'The movie was okish.',
        'The movie was terrible...'
    ]

    main(saved_model_path, test_texts)


