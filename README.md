# mnist_permutations

Next steps

Compare model accuracy on unpermuted vs. permuted columns

Assuming that permuting the columns degrades accuracy, write a dataloader to create a dataset for 
permuted columns/ordering, with the following caveats:

- If a column is blank move it to the (order=28)

- Number remaining columns 0..x

- Permute - with target values equal to index
