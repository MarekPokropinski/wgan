from unittest import TestCase, main

import tensorflow as tf

from models import MnistGenerator, MnistDiscriminator

class TestMnist(TestCase):
    def test_generator_output(self):
        inp = tf.random.normal(shape=(1, 128))
        model = MnistGenerator()
        model_output = model(inp)

        self.assertEqual((1, 28, 28, 1), model_output.shape)

    def test_generator_output(self):
        inp = tf.random.normal(shape=(1, 28, 28, 1))
        model = MnistDiscriminator()
        model_output = model(inp)
        self.assertEqual((1, 1), model_output.shape)

if __name__=="__main__":
    main()
