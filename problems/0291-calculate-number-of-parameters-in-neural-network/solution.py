def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    total = 0

    for layer in layers:

        # Dense layer
        if layer["type"] == "dense":
            num_params = (
                layer["input_size"] *
                layer["output_size"]
            )

            if layer["use_bias"]:
                num_params += layer["output_size"]

            total += num_params

        # Conv2D layer
        elif layer["type"] == "conv2d":
            kernel_size = layer["kernel_size"]

            if isinstance(kernel_size, int):
                kernel_h = kernel_size
                kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size

            num_params = (
                kernel_h *
                kernel_w *
                layer["in_channels"] *
                layer["out_channels"]
            )

            if layer["use_bias"]:
                num_params += layer["out_channels"]

            total += num_params

        # Embedding layer
        elif layer["type"] == "embedding":
            num_params = (
                layer["num_embeddings"] *
                layer["embedding_dim"]
            )

            total += num_params

        else:
            raise ValueError(f"Unsupported layer type: {layer['type']}")

    return total

    