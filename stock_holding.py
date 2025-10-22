class StockHolding:
    """
    A class to represent a holding of a sotkc in the portfolio.
    It tracks the stock ticker, total quantity of shares, and the average purchase price.
    """
    def __init__(self, symbol: str, quantity: int, price: float):
        if quantity < 0:
            raise ValueError("quantity cannot be negative.")
        if price < 0:
            raise ValueError("price cannot be negative.")
        self.symbol = symbol.upper()
        self.quantity = int(quantity)
        #Average price per share 
        self.avg_price = float(price)

    def add_shares(self, quantity: int, price: float) -> None:
        """
        Add shares of the stock to this holding at the given price.
        The average price is recalculated as a weighted average.
        """
        if quantity <= 0:
            return
        total_cost_before = self.avg_price * self.quantity
        additional_cost = price * quantity
        new_quantity = self.quantity + quantity
        # new average = total cost / total shares
        if new_quantity > 0:
            self.avg_price = (total_cost_before + additional_cost) / new_quantity
        self.quantity = new_quantity
    def remove_shares(self, quantity: int):
        """
        Remove shares from the holding (e.g., after selling).
        Returns True if all shares were sold (quantity == 0 ), False otherwise.
        """
        if quantity <= 0:
            #selling all shares
            self.quantity = 0
            return True
        else:
            self.quantity -= quantity
            return False
    def __repr__(self):
        return f"StockHolding(symbol={self.symbol}, quantity={self.quantity}, avg_price={self.avg_price:.2f})"