def relu(input):
    output = max(0, input)
    
    return(output)

node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

hidden_layer_outputs = np.array([node_0_output, node_1_output])

model_output = (hidden_layer_outputs * weights['output']).sum()

print(model_output)
