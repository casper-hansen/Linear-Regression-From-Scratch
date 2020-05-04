# define our data with manual coefficients
intercept = 3.4

coeffs = [3.2, 2.1, 4.5, 2.6]

dataset = {
    0: [3, 1.5],
    1: [1, 7.6],
    2: [3, 8.3],
    3: [4, 3.2]
}

def assemble_equation(intercept, coefficients, x_values=None, output=None):
    equation = f'y = {intercept}'
    for i, coef in enumerate(coefficients):
        equation += ' '
        equation += '+'
        equation += ' '
        if x_values == None:
            equation += f'x{i+1}'
        else:
            equation += f'{x_values[i]}'
        equation += '*'
        equation += str(coef)

    if output != None:
        equation += f' = {output}'
    return equation

def calculate_output(intercept, coefficients, dataset):
    # check for missing values, i.e. that the number of observations for each feature is the same
    equal_length = {key: len(value) for key, value in dataset.items()}
    assert len(set([val for val in equal_length.values()])) == 1

    # number of observations = number of outputs
    n_outputs = equal_length.get(0)

    x_values = [[None for i in range(n_outputs)] for n in range(len(dataset))]

    for key, array in dataset.items():
        x_index = x_values[key]
        for i in range(len(x_index)):
            x_index[i] = array[i]

    # transpose x_values
    x_values = list(zip(*x_values))
    
    outputs = []

    for i in range(n_outputs):
        result = 0
        for j, coef in enumerate(coefficients):
            result += x_values[i][j] * coef
        result += intercept

        outputs.append(result)

    return outputs, x_values

# get the formula
formula = assemble_equation(intercept, coeffs)

# calculate outputs from x_values
outputs, x_values = calculate_output(intercept, coeffs, dataset)

print(f'Formula: {formula}')

# display the equations for each observation
for i, output in enumerate(outputs):
    eq = assemble_equation(intercept, coeffs, x_values[i], output)
    print(f'Equation {i+1}: {eq}')