import numpy as np

def sigmoid(x):
    # Функция активации: сигмоид
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Сумматор: взвешенная сумма весов и входов + смещение (net)
        total = np.dot(self.weights, inputs) + self.bias
        # Функция активации
        return sigmoid(total)

# Тестирование нейрона
if __name__ == "__main__":
    weights = np.array([0, 1])
    bias = 4
    neuron = Neuron(weights, bias)
    x = np.array([2, 3])
    print("Вывод нейрона:", neuron.feedforward(x))
    # Ожидаемый результат: 0.999
