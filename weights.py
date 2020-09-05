def get_weights(net, path_to_save=None):
    is_first = True
    row_vector = None
    
    for param in net.parameters():
        if is_first:
            row_vector = param.data.reshape(1, -1)
            is_first = False
        else:
            row_vector = torch.cat((row_vector, param.data.reshape(1, -1)), 1)
    
    if path_to_save != None:
        torch.save(row_vector, path_to_save)
        
    return row_vector
	
	
def set_weights(net, row_vector):
    passed = 0
    
    for param in net.parameters():
        num_parameters = param.data.reshape(1, -1).shape[1]
        weights = row_vector[0][passed : passed + num_parameters].reshape(param.data.shape)
        
        with torch.no_grad():
            param.data = weights
        
        passed += num_parameters