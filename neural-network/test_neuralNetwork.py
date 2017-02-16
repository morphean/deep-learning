from unittest import TestCase, main
from neuralNetwork import NeuralNetwork

class TestNeuralNetwork(TestCase):
    # def test_train(self):
    #     self.fail()
    #
    # def test_predict(self):
    #     self.fail()

    def test_sigmoid(self):
        result = NeuralNetwork().sigmoid(1)

        self.assertEqual(result, 0.7310585786300049)

    def test_sigmoidDeriavative(self):
        result = NeuralNetwork().sigmoid_deriavative(0.7310585786300049)
        self.assertEqual(result, 0.19661193324148185)

if __name__ == '__main__':
    main()