## MEV Contest

[sauce](https://alphamev.ai)

Predict back-runable transactions and cumulative miner's profit that this
transaction would generate. There are many examples of transactions which open
MEV opportunities after them:

1) Oracle updates allow to perform liquidations.
2) Large AMM swaps allow to perform cross-DEX arbitrage.
3) Accepted govenance proposals which change pool parameters.
And many others.

Each row of the training dataset contains following columns:

1) txHash - transaction hash on Ethereum blockchain
2) txData - dictionary representing all basic transaction information
3) txTrace - Geth-style transaction trace
4) Label0 - Binary label whether this transaction is back-runable.
5) Label1 - Total amount of ETH sent to miners as bribes via MEV-bundles due to this transaction.

You can find link to the dataset below, it's a zip archive containing 2 files: "train.csv" and "test.csv".
For each row in "test.csv" you're expected to generate two predictions separated by comma:
1) P[Label0 == 1]
2) E[Label1 | Label0 == 1]
You can also find most basic solution in Python which generates required predictions in correct format using the link below.
