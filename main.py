#IMAGE RECOGNITION
#Stoicastic gradient descent
#version 1.0

import numpy as np
import time
import string
import pygame
from pygame.locals import *

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()

#todo:
#add input cache dataset with stretched, distorted shapes
#add output cache

#import the network functions from another file (so it acts like a library)

def main():

    
    screen.fill("blue")

    lines = np.array([])

    #train then pygame loop
    running = True
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if len(lines) == 0:
                    lines = np.array([mouse_pos])
                else:
                    lines = np.vstack((lines, mouse_pos))
            elif event.type == K_e:
                screen.fill("blue")

        # fill the screen with a color to wipe away anything from last frame

        screen.fill("BLACK")
        draw_lines(lines)
        # RENDER YOUR GAME HERE

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

def draw_lines(array):
    if len(array) % 2 == 0 and not len(array) == 0:
        for i in range(int(len(array) / 2)):
            pygame.draw.line(screen, "WHITE", array[2 * i], array[2 * i + 1], 10)
    elif len(array) % 2 == 1 and not len(array) == 1:
        for i in range(int((len(array) - 1) / 2)):
            pygame.draw.line(screen, "WHITE", array[2 * i], array[2 * i + 1], 10)
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.line(screen, "BLUE", array[-1], mouse_pos, 10)
    elif len(array) == 1:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.line(screen, "BLUE", array[-1], mouse_pos, 10)




    # input_cache = np.array([[-5],[-4],[-3],[-2],[-1],[0],[1],[2],[3],[4],[5]])
    # output_cache = input_cache ** 2

    # met = NETWORK((1,10,1), activation="SIGMOID", IAFLL=False)

    # met.train(input_cache, output_cache,epochs=5000, lr=0.01)

    # met.test(input_cache, output_cache)

    # met.desmos_format1D()

class NETWORK:
    #Values represent default config
    def __init__(self, size=(1,1), loss="MEAN_SQUARED_ERROR", activation="NONE", IAFLL=False):
        self.size = size
        self.loss = loss
        self.Layers = architecture(size, loss, activation)
        self.activation_function = activation
        self.Include_activation_for_last = IAFLL

    def train(self, input_cache, output_cache, epochs=1000, lr=0.05, optimizer="STOICHASTIC_GRADIENT_DECSENT"):
        print(f"---------TRAINING----------")
        if optimizer == "STOICHASTIC_GRADIENT_DECSENT":
            for epoch in range(epochs):
                for index in range(len(input_cache)):
                    inputs = input_cache[index]
                    outputs = output_cache[index]

                    #foward
                    self.foward_pass(inputs, outputs)

                    #backward
                    self.backprop(outputs)

                    #update parameters
                    self.update_parameters(lr)
        elif optimizer == "GRADIENT_DECSENT":
            return
        elif optimizer == "MOMENTUM":
            #momentum gradient descent here
            return
        else:
            print("Please enter an optimizer for training")
            return
        print(f"-----FINISHED-----")

    def foward_pass(self, inputs, outputs):
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)

        if self.Include_activation_for_last:
            self.Layers[-1].foward(self.Layers[-2].outputs, outputs)
        else:
            self.Layers[-1].foward(self.Layers[-2].weighted_sums, outputs)

    def foward_return(self, inputs, outputs):
        x = inputs.reshape(-1,1)
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                x = layer.foward(x)

        predicted = None
        #insert softmax logic here
        if self.Include_activation_for_last:
            predicted = self.Layers[-2].outputs
        else:
            predicted = self.Layers[-2].weighted_sums
        
        cost = self.Layers[-1].foward(predicted, outputs)
        predicted = self.Layers[-1].predicted
        return cost, predicted

    def telementary(self):
        for layer in self.Layers:
            if layer.type == "Dense":
                layer.telementary()

    def backprop(self, outputs):
        following_layer = self.Layers[-1]
        following_layer.backprop(outputs)
        
        for index, layer in enumerate(reversed(self.Layers)):
            if not index == 0:
                layer.backprop(following_layer, self.Include_activation_for_last)
                following_layer = layer

    def update_parameters(self, lr):
        for index, layer in enumerate(self.Layers):
            if index < len(self.Layers) - 1:
                layer.update_parameters(lr)

    def test(self, input_cache, output_cache, telementary=True):
        print("\nTEST:")
        total_cost = 0
        best_case_index = 0
        worst_case_index = 0
        best_case_cost = 1000
        worst_case_cost = 1000
        for index, inputs in enumerate(input_cache):
            outputs = output_cache[index]
            cost, predicted = self.foward_return(inputs, outputs)
            total_cost += cost

            if index == 0:
                best_case_cost = cost
                worst_case_cost = cost
            if worst_case_cost < cost:
                worst_case_index = index
                worst_case_cost = cost
            if best_case_cost > cost:
                best_case_index = index
                best_case_cost = cost

        if worst_case_index == best_case_index:
            if telementary:
                outputs = output_cache[worst_case_index]
                inputs = input_cache[worst_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Predicted:\n", predicted)
                print("Actual:\n", outputs.reshape(-1,1))
        else:
            if telementary:
                outputs = output_cache[worst_case_index]
                inputs = input_cache[worst_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Worst Case Predicted:\n", predicted)
                print("Worst Case Actual:\n", outputs.reshape(-1,1))

                outputs = output_cache[best_case_index]
                inputs = input_cache[best_case_index]
                cost, predicted = self.foward_return(inputs, outputs)

                print("Best Case Predicted:\n", predicted)
                print("Best Case Actual:\n", outputs.reshape(-1,1))

            print(f"Worst Case: {worst_case_cost:.6f}\nBest Case: {best_case_cost:.6f}")
        print(f"Total Cost: {total_cost:.4f}")

    def desmos_format1D(self, input = "x"):
        if not self.Layers[0].num_inputs == 1:
            print("\n******Desmos 1D only works with one input******")
            return

        abc = list(string.ascii_uppercase) + list(string.ascii_lowercase)
        abc.remove(input)
        abc.remove("e")
        
        if self.number_of_nuerons() > len(abc):
            print("\n******Desmos has too little functions to represent network******")
            return

        print("\nDESMOS COPY PASTE BELOW:", end="")

        func = []
        prev_func = []

        func.append(abc[0])
        abc.pop(0)
        if self.activation_function == "NONE":
            print(f"\n{func[0]}({input})={input}")
        elif self.activation_function == "RELU":
            print(f"\n{func[0]}({input})=max({input},0)")
        elif self.activation_function == "SIGMOID":
            print(f"\n{func[0]}({input})=1/(1+exp(-{input}))")
        elif self.activation_function == "TANH":
            print(f"\n{func[0]}({input})=tanh({input})")
        
        for index, layer in enumerate(self.Layers):
            temp = []
            if (not layer.type == "Cost") and (index == 0):
                for i, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)
                    m = weight[0]
                    b = layer.biases[i][0]

                    
                    if b >= 0:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}+{b:.5f})")
                    else:
                        print(f"{func[-1]}({input})={func[0]}({m:.5f}{input}{b:.5f})")
                prev_func.append(temp)
            elif not layer.type == "Cost":
                for i, weight in enumerate(layer.weights):
                    func.append(abc[0])
                    temp.append(abc[0])
                    abc.pop(0)

                    b = layer.biases[i][0]
                    equation = ""
                    for i, m in enumerate(weight):
                        if m >= 0:
                            equation += f"+{m:.5f}{prev_func[index - 1][i]}({input})"
                        else:
                            equation += f"{m:.5f}{prev_func[index - 1][i]}({input})"
                    if b >= 0:
                        if index == len(self.Layers) - 2 and not self.Include_activation_for_last:
                            print(f"{func[-1]}({input})={equation}+{b:.5f}")
                        else:
                            print(f"{func[-1]}({input})={func[0]}({equation}+{b:.5f})")
                    else:
                        if index == len(self.Layers) - 2 and not self.Include_activation_for_last:
                            print(f"{func[-1]}({input})={equation}{b:.5f}")
                        else:
                            print(f"{func[-1]}({input})={func[0]}({equation}{b:.5f})")
                prev_func.append(temp)

    def number_of_nuerons(self):
        num_nuerons = 0
        for layer in self.Layers:
            num_nuerons += layer.num_outputs
        return num_nuerons

class DENSE_LAYER:
    def __init__(self, num_inputs, num_outputs, activation_type):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_outputs, num_inputs) * np.sqrt(2 / num_inputs)
        self.biases = np.random.randn(num_outputs, 1) * np.sqrt(2 / num_inputs)
        self.activation = activation_type
        self.type = "Dense"

    def foward(self, inputs):
        self.inputs = inputs
        z = self.weights @ inputs
        z += self.biases
        a = None
        if self.activation == "RELU":
            a = np.maximum(z, 0)
        elif self.activation == "SIGMOID":
            a =  1 / (1 + np.exp(-np.maximum(np.minimum(z,15),-15)))
        elif self.activation == "TANH":
            a = np.tanh(z)
        elif self.activation == "NONE":
            a = z
        self.weighted_sums = z
        self.outputs = a
        return a

    def telementary_parameters(self):
        print("\nweights: ")
        print(self.weights)
        print("\nbiases: ")
        print(self.biases)

    def backprop(self, following_layer, IAFLL):
        aft_deltas = following_layer.deltas
        #update
        if self.activation == "RELU":
            partial_derivative = (self.weighted_sums > 0).astype(dtype=float)
        elif self.activation == "SIGMOID":
            partial_derivative = self.outputs * (1 - self.outputs)
        elif self.activation == "TANH":
            partial_derivative = 1 - np.tanh(self.weighted_sums) ** 2
        elif self.activation == "NONE":
            partial_derivative = 1

        if not following_layer.type == "Cost":
            aft_weights = following_layer.weights
            self.deltas = np.transpose(aft_weights) @ aft_deltas * partial_derivative     
        else:
            if IAFLL:
                self.deltas = aft_deltas * partial_derivative
            else:
                self.deltas = aft_deltas
        
    def update_parameters(self, lr):
        self.biases = self.biases - lr * self.deltas
        self.weights = self.weights - lr * (self.deltas @ self.inputs.T)

class COST_LAYER:
    def __init__(self, cost_type):
        self.cost = cost_type
        self.type = "Cost"
        self.num_outputs = 0

    def foward(self, inputs, actual):
        self.predicted = inputs
        c = None
        if self.cost == "MEAN_SQUARED_ERROR":
            c = np.sum((actual.reshape(-1,1) - inputs) ** 2)
        elif self.cost == "BINARY_CROSS_ENTROPY":
            c = binary_cross_entropy(inputs, actual.reshape(-1,1))
        elif self.cost == "BINARY_CROSS_ENTROPY_AND_SOFTMAX":
            self.predicted = stable_softmax(inputs)
            c = binary_cross_entropy(stable_softmax(inputs), actual.reshape(-1,1))
        self.outputs = c
        return c

    def backprop(self, actual):
        if self.cost == "MEAN_SQUARED_ERROR":
            self.deltas = 2 * (self.predicted - actual.reshape(-1,1))
        elif self.cost == "BINARY_CROSS_ENTROPY":
            self.deltas = -1 * (actual.reshape(-1,1) / self.predicted - (1-actual.reshape(-1,1))/(1-self.predicted))
        elif self.cost == "BINARY_CROSS_ENTROPY_AND_SOFTMAX":
            # A=np.arange(16).reshape(4,4)
            # print(A)
            # print(A * (-np.eye(4,4) + 1))
            # print(A * (-np.eye(4,4) + 1) + np.identity(4))
            sfmx = stable_softmax(self.predicted)
            dcdp = -(actual.reshape(-1,1) / sfmx - (1 - actual.reshape(-1,1))/(1 - sfmx))
            eq = sfmx * (1 - sfmx)      
            uneq = (sfmx @ np.transpose(sfmx)) * (-np.eye(sfmx.size) + 1) + np.eye(sfmx.size)  
            dpda = eq + np.sum(uneq, axis=1).reshape(-1,1)
            self.deltas = dcdp * dpda

def architecture(size, loss, activation_type):
    Layers = []
    for index in range(len(size) - 1):
        Layers.append(DENSE_LAYER(size[index], size[index + 1], activation_type))
    Layers.append(COST_LAYER(loss))
    return Layers

def stable_softmax(matrix):
    shiftx = matrix - np.max(matrix)
    mat = np.exp(shiftx)
    return mat / np.sum(mat)

def cross_entropy(matrix_p, matrix_q):
    if not matrix_p.shape == matrix_q.shape:
        print("Cross Entopy shape mismatch")
        return False

    mat = np.multiply(matrix_p, np.log(matrix_q))
    return np.minimum(-np.sum(mat), 1000)

def binary_cross_entropy(matrix_p, matrix_q):
    if not matrix_p.shape == matrix_q.shape:
        print("Cross Entopy shape mismatch")
        return False
    
    #have to use np.nan_to_num because network arrives at log(0)
    matlog1 = np.nan_to_num(np.log(matrix_p), nan=-10000)
    matlog2 = np.nan_to_num(np.log(1 - matrix_p), nan=-10000)

    mat = matrix_q * matlog1 + (1 - matrix_q) * matlog2
    return np.minimum(-np.sum(mat), 1000)

def normalize(numpy_array):
    return numpy_array / max(numpy_array)

start = time.perf_counter()
print("\n---------------RUNNING---------------\n")
main()
end = time.perf_counter()
print(f"\n-----FINISHED--IN--{(end - start):.9f}s------\n")

pygame.quit()
