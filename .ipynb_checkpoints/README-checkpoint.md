# topoformer-ssm
Implementing Topoformer Architecture on State-Space Models. Project in collaboration with Georgia Tech and Harvard University.

## Training

### Dimensionality Parameters & Scaling Rules: 

1. d_model:
   - This can be changed relatively freely, but it's common to use powers of 2 (e.g., 128, 256, 512, 1024).
   - Changing d_model will affect the size of many weight matrices in the model, so it has a significant impact on the total number of parameters.

2. d_state:
   - This is typically smaller than d_model.
   - Common values are 16, 32, 64, or 128.
   - It should be chosen such that d_state <= d_model.

3. d_conv:
   - This represents the kernel size of the 1D convolution.
   - Typical values are 3, 4, or 5.
   - Larger values increase the receptive field but also computational cost.
   - There's no strict mathematical constraint, but very large values (e.g., > 7) are uncommon and may not provide much benefit.

4. expand:
   - This is typically an integer >= 1.
   - Common values are 2 or 4.
   - It determines the expansion factor for the inner dimension: d_inner = d_model * expand
   - Larger values increase model capacity but also memory usage.

5. num_layers:
   - This can be changed freely without causing matrix multiplication issues.
   - More layers generally mean more capacity, but also more computation and potential for training difficulties (e.g., vanishing gradients).
   - Common values range from 4 to 32, depending on the task complexity and available computational resources.

#### General rules and considerations:

1. Consistency: Ensure that d_state <= d_model to avoid dimension mismatches.

2. Model capacity: The total model capacity is influenced by the product of these parameters. If you increase one, you might be able to decrease another while maintaining similar capacity.

3. Computational constraints: Larger values for any of these parameters will increase memory usage and computation time. Consider your hardware limitations.

4. Powers of 2: For d_model and d_state, using powers of 2 can sometimes lead to more efficient computation on GPUs.

5. Balanced scaling: When increasing model size, it's often beneficial to increase multiple parameters together rather than scaling just one extremely high.

6. Start small: Begin with smaller values and gradually increase them while monitoring performance improvements.

#### A sample progression of configurations might look like this:

1. Small: d_model=256, d_state=32, d_conv=4, expand=2, num_layers=4
2. Medium: d_model=512, d_state=64, d_conv=4, expand=2, num_layers=8
3. Large: d_model=1024, d_state=128, d_conv=5, expand=4, num_layers=16

Remember, the best configuration often depends on your specific task and dataset. It's a good practice to experiment with different configurations and use techniques like grid search or Bayesian optimization to find the best hyperparameters for your particular use case.