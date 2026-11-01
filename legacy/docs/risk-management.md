# Historical execution assumptions

Author: Salvatore (Spirax-M1).

The archived bots contain position sizing, stop/target handling and broker-specific controls. Their presence does not establish that the complete application was reliable. The historical simulator uses simplified fills and currency/contract assumptions; V26 explicitly omits commission and swap.

The supported offline pipeline makes no live risk-management claim. It simulates one unlevered position per execution bar, subtracts a fixed round-trip cost and measures drawdown at bar closes. It does not implement a broker, margin, financing, liquidity constraints or intrabar drawdown.

The old deployment guidance and suggested capital allocations have been removed. See [methodology](../../docs/methodology.md) for the actual calculation contract and [audit](../../docs/audit.md) for unresolved historical defects.

Original snapshot: [source and previous documentation](https://github.com/Spirax-M1/algorithmic-trading-ml/tree/930e3aa552236727e24ea68f9cec5280d2bd5ed8).
